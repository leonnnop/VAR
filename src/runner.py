'''
This script handles the training process.
'''

import os
import math
import time
import json
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.model import get_model
from src.model.utils.optimization import BertAdam, EMA
from src.dataloader.var_dataset import VARDataset, sentences_collate

from src.engine.train import train_epoch
from src.engine.valid import eval_epoch
from src.engine.translate import translate_w_metrics

from src.utils import save_parsed_args_to_json, is_distributed, init_seed, create_save_dir, dist_log


class runner():

    @staticmethod
    def train_logic(epoch_i, model, training_data, optimizer, ema, device, opt):
        if is_distributed():
            training_data.sampler.set_epoch(epoch_i)
            dist_log('Setting sampler seed: {}'.format(training_data.sampler.epoch))

        start = time.time()
        if ema is not None and epoch_i != 0:  # * use normal parameters for training, not EMA model
            ema.resume(model)
        train_loss, train_acc = train_epoch(
            model, training_data, optimizer, ema, device, opt, epoch_i)

        dist_log('[Training]  ppl: {ppl: 8.5f}, accuracy: {acc:3.3f} %, elapse {elapse:3.3f} min'
                    .format(ppl=math.exp(min(train_loss, 100)), acc=100*train_acc, elapse=(time.time()-start)/60.))

        # * Note here GT words are used to predicted next words, the same as training case!
        if ema is not None:
            ema.assign(model)  # * EMA model

        return model, ema, train_loss, train_acc

    @staticmethod
    def eval_logic(model, validation_data, device, opt):
        start = time.time()
                
        val_loss, val_acc = eval_epoch(model, validation_data, device, opt)

        dist_log('[Val]  ppl: {ppl: 8.5f}, accuracy: {acc:3.3f} %, elapse {elapse:3.3f} min'
                    .format(ppl=math.exp(min(val_loss, 100)), acc=100*val_acc, elapse=(time.time()-start)/60.))

        return model, val_loss, val_acc

    @staticmethod
    def translate_logic(epoch_i, model, translation_data, opt, prev_best_score):          

        # * Note here we use greedy generated words to predicted next words, the true inference situation.
        if hasattr(model, 'module'):
            _model = model.module
        else: _model = model

        checkpoint = {
            'model': _model.state_dict(),  # * EMA model
            'model_cfg': _model.config,
            'epoch': epoch_i}

        val_greedy_output, filepaths = translate_w_metrics(
                checkpoint, translation_data, opt, eval_mode=opt.evaluate_mode, model=_model)

        if opt.local_rank <= 0:
            if 'BertScore' not in val_greedy_output:
                val_greedy_output['BertScore'] = [0.]

            dist_log('[Val] METEOR {m:.2f} Bleu@4 {b:.2f} CIDEr {c:.2f} ROUGE_L {r:.2f} BERT_S {s:.2f}'
                        .format(m=val_greedy_output['METEOR'][0]*100,
                                b=val_greedy_output['Bleu_4'][0]*100,
                                c=val_greedy_output['CIDEr'][0]*100,
                                r=val_greedy_output['ROUGE_L'][0]*100,
                                s=val_greedy_output['BertScore'][0]*100))

            if opt.save_mode == 'all':
                model_name = opt.save_model + '_e{}.chkpt'.format(epoch_i)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if val_greedy_output['CIDEr'][0] > prev_best_score:
                    prev_best_score = val_greedy_output['CIDEr'][0]
                    torch.save(checkpoint, model_name)
                    new_filepaths = [e.replace('tmp', 'best') for e in filepaths]
                    for src, tgt in zip(filepaths, new_filepaths):
                        os.renames(src, tgt)
                    dist_log('The checkpoint file has been updated.')  

        return model, val_greedy_output, prev_best_score

    @staticmethod
    def run(model, training_data, validation_data, translation_data, device, opt):

        # * Prepare optimizer
        param_optimizer = list(model.named_parameters())
            
        if opt.use_shared_txt_emb:
            for parameter in model.module.embeddings_dec2.txt_emb.parameters():
                parameter.requires_grad = False

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if (p.requires_grad and (not any(nd in n for nd in no_decay)))], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if (p.requires_grad and any(nd in n for nd in no_decay))], 'weight_decay': 0.0}
        ]

        if opt.ema_decay != -1:
            ema = EMA(opt.ema_decay)
            for name, p in model.named_parameters():
                if p.requires_grad:
                    ema.register(name, p.data)
        else: ema = None

        num_train_optimization_steps = len(training_data) * opt.n_epoch
        optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=opt.lr,
                            warmup=opt.lr_warmup_proportion,
                            t_total=num_train_optimization_steps,
                            schedule='warmup_linear')

        log_train_file = None
        log_valid_file = None
        if opt.log and opt.local_rank <= 0:
            log_train_file = opt.log + '.train.log'
            log_valid_file = opt.log + '.valid.log'

            dist_log('Training performance will be written to file: {} and {}'.format(
                log_train_file, log_valid_file))

            with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
                log_tf.write('epoch,loss,ppl,accuracy\n')
                log_vf.write('epoch,loss,ppl,accuracy,METEOR,BLEU@4,CIDEr,ROUGE\n')

        prev_best_score = 0. 
        for epoch_i in range(opt.n_epoch):
        
            dist_log('[Epoch {}]'.format(epoch_i))

            model, ema, train_loss, train_acc = \
                    runner.train_logic(epoch_i, model, training_data, optimizer, ema, device, opt)

            model, val_loss, val_acc = runner.eval_logic(model, validation_data, device, opt)

            if epoch_i >= opt.trans_sta_epoch:
                model, val_greedy_output, prev_best_score = runner.translate_logic(epoch_i, model, translation_data, opt, prev_best_score)

                if opt.local_rank <= 0:
                    cfg_name = opt.save_model +'.cfg.json'
                    save_parsed_args_to_json(opt, cfg_name)

                    if log_train_file and log_valid_file:
                        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{acc:3.3f}\n'.format(
                                epoch=epoch_i, loss=train_loss, ppl=math.exp(min(train_loss, 100)), acc=100*train_acc))
                            log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{acc:3.3f},{m:.2f},{b:.2f},{c:.2f},{r:.2f},{s:.2f}\n'.format(
                                epoch=epoch_i, loss=val_loss, ppl=math.exp(min(val_loss, 100)), acc=100*val_acc,
                                m=val_greedy_output['METEOR'][0]*100,
                                b=val_greedy_output['Bleu_4'][0]*100,
                                c=val_greedy_output['CIDEr'][0]*100,
                                r=val_greedy_output['ROUGE_L'][0]*100,
                                s=val_greedy_output['BertScore'][0]*100
                                ))
                    
        return model
        

def get_args():
    '''parse and preprocess cmd line args'''
    parser = argparse.ArgumentParser()

    # * basic settings
    parser.add_argument('--dset_name', type=str, default='VAR', choices=['VAR'],
                        help='Name of the dataset, will affect data loader, evaluation, etc')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--data_dir', required=True, help='dir containing the splits data files')
    parser.add_argument('--res_root_dir', type=str, default='results', help='dir to containing all the results')

    # * training config -- batch/lr/eval etc.
    parser.add_argument('--n_epoch', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--val_batch_size', type=int, default=16, help='inference batch size')
    parser.add_argument('--trans_batch_size', type=int, default=16, help='tranlating batch size')
    parser.add_argument('--lr', type=float, default=16e-5)
    parser.add_argument('--lr_warmup_proportion', default=0.1, type=float,
                        help='Proportion of training to perform linear learning rate warmup for. '
                             'E.g., 0.1 = 10% of training.')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Use soft target instead of one-hot hard target')
    parser.add_argument('--grad_clip', type=float, default=1, help='clip gradient, -1 == disable')
    parser.add_argument('--ema_decay', default=0.9999, type=float,
                        help='Use exponential moving average at training, float in (0, 1) and -1: do not use.  '
                             'ema_param = new_param * ema_decay + (1-ema_decay) * last_param')

    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--no_pin_memory', action='store_true',
                        help='Don\'t use pin_memory=True for dataloader. '
                             'ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num subprocesses used to load the data, 0: use main process')
    parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best',
                        help='all: save models at each epoch; best: only save the best model')
    parser.add_argument('--no_cuda', action='store_true', help='run on cpu')

    parser.add_argument('--evaluate_mode', type=str, default='test', choices=['val', 'test']) 
    parser.add_argument('--score_mode', type=str, default='event', choices=['event']) 
    parser.add_argument('--trans_sta_epoch', type=int, default=2)       

    # * model overall config
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--intermediate_size', type=int, default=768)
    parser.add_argument('--vocab_size', type=int, help='number of words in the vocabulary')
    parser.add_argument('--word_vec_size', type=int, default=300)
    parser.add_argument('--video_feature_size', type=int, default=3072, help='2048 appearance + 1024 flow')
    parser.add_argument('--max_v_len', type=int, default=50, help='max length of video feature')
    parser.add_argument('--max_t_len', type=int, default=22,
                        help='max length of text (sentence or paragraph)')
    parser.add_argument('--max_n_len', type=int, default=12,
                        help='for recurrent, max number of sentences')

    # * initialization config
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='number of transformer layers')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.1)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--initializer_range', type=float, default=0.02)
                        
    # * visual encoder config
    parser.add_argument('--K', type=int, default=4, help='number of cascades')
    parser.add_argument('--loss_aux_weight', type=float, default=0.1, help='')
    parser.add_argument('--momentum_aux_m', type=float, default=0.999)   

    # * linguistic encoder config
    parser.add_argument('--glove_path', type=str, default=None, help='extracted GloVe vectors')
    parser.add_argument('--glove_version', type=str, default=None, help='extracted GloVe vectors')
    parser.add_argument('--freeze_glove', action='store_true', help='do not train GloVe vectors')
    parser.add_argument('--share_wd_cls_weight', action='store_true',
                        help='share weight matrix of the word embedding with the final classifier, ')

    # * cascade decoder config
    parser.add_argument('--num_dec1_blocks', type=int, default=4)
    parser.add_argument('--num_dec2_blocks', type=int, default=4)
    parser.add_argument('--loss_aux_caption', type=float, default=0.25)  
    parser.add_argument('--loss_main_caption', type=float, default=0.25)  
    parser.add_argument('--use_shared_txt_emb', action='store_true')
    # * scheduled sampling
    parser.add_argument('--disable_scheduled_sampling', action='store_true')
    parser.add_argument('--scheduled_method', type=str, default='probability', choices=['confidence', 'probability'])
    parser.add_argument('--conf_replace_tr', type=float, default=0.5)
    parser.add_argument('--decay_k', type=int, default=10) 
    # * sentence embedding and conf hyper-parameters
    parser.add_argument('--sentence_emb_aggregation_mode', type=str, default='max', choices=['mean', 'max', 'weighted'])
    parser.add_argument('--conf_bucket_size', type=int, default=10) 
    parser.add_argument('--conf_temperature', type=float, default=1.0) 

    # * post process and compatibility check
    opt = parser.parse_args()
    opt.local_rank = int(os.environ['LOCAL_RANK']) 
    opt.cuda = not opt.no_cuda 
    opt.pin_memory = not opt.no_pin_memory

    if opt.share_wd_cls_weight:
        assert opt.word_vec_size == opt.hidden_size, \
            'hidden size has to be the same as word embedding size when ' \
            'sharing the word embedding weight and the final classifier weight'

    if opt.K==1: 
        opt.loss_main_caption = 1. 
        opt.loss_aux_caption = 0. 
        opt.disable_scheduled_sampling = True

    return opt  


def main():
    opt = get_args()

    init_seed(opt.seed, cuda_deterministic=True)

    train_dataset = VARDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir,
        max_t_len=opt.max_t_len, max_v_len=opt.max_v_len, max_n_len=opt.max_n_len, 
        mode='train', K=opt.K, word2idx_path=os.path.join(opt.glove_path, opt.glove_version.replace('vocab_glove', 'word2idx').replace('.pt', '.json'))
        )
    val_dataset = VARDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir,
        max_t_len=opt.max_t_len, max_v_len=opt.max_v_len, max_n_len=opt.max_n_len, 
        mode=opt.evaluate_mode, K=opt.K, word2idx_path=os.path.join(opt.glove_path, opt.glove_version.replace('vocab_glove', 'word2idx').replace('.pt', '.json'))
        )

    distributed = opt.local_rank >= 0
    if distributed:
        device = torch.device('cuda:{}'.format(opt.local_rank))    
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://',
        )
    else: device = torch.device('cuda' if opt.cuda else 'cpu')

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
    else: train_sampler, val_sampler = None, None

    opt = create_save_dir(opt)
    
    train_loader = DataLoader(train_dataset, collate_fn=sentences_collate,
                              batch_size=opt.batch_size, 
                              shuffle=(train_sampler is None),
                              num_workers=opt.num_workers, pin_memory=opt.pin_memory,
                              sampler=train_sampler)

    val_loader = DataLoader(val_dataset, collate_fn=sentences_collate,
                            batch_size=opt.val_batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=opt.pin_memory,
                            sampler=val_sampler)

    trans_loader = DataLoader(val_dataset, collate_fn=sentences_collate,
                            batch_size=opt.trans_batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=opt.pin_memory,
                            sampler=val_sampler)

    opt.vocab_size = len(train_dataset.word2idx)
    if opt.local_rank <= 0: print(json.dumps(vars(opt), indent=4, sort_keys=True))

    model = get_model(opt)

    if opt.glove_path is not None:
        if hasattr(model, 'embeddings_decs'):
            dist_log('Load GloVe as word embedding')
            for k in range(opt.K):
                model.embeddings_decs[k].txt_emb.set_pretrained_embedding(
                    torch.from_numpy(torch.load(os.path.join(opt.glove_path, opt.glove_version))).float(), freeze=opt.freeze_glove)
        else: dist_log('[WARNING] This model has no embeddings, cannot load glove vectors into the model')

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank
        )
    else: model = model.to(device)

    runner().run(model, train_loader, val_loader, trans_loader, device, opt)
    

if __name__ == '__main__':
    main()
