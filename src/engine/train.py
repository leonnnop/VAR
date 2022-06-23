
import torch
import torch.nn as nn

from src.dataloader import cal_performance, prepare_batch_inputs, VARDataset
from src.utils import is_distributed, reduce_tensor, get_world_size, dist_log

def train_epoch(model, training_data, optimizer, ema, device, opt, epoch):
    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    if not opt.disable_scheduled_sampling:
        model.module.set_epoch(epoch)

    for batch_idx, batch in enumerate(training_data):
        niter = epoch * len(training_data) + batch_idx
        
        # * prepare data
        batched_data = prepare_batch_inputs(batch[0], device=device, non_blocking=opt.pin_memory)
        
        num_sen = batched_data['num_sen']
        # * decoders
        input_labels = batched_data['gt']
        input_ids = batched_data['decoder_input']['input_ids']
        input_masks = batched_data['decoder_input']['input_mask']
        token_type_ids = batched_data['decoder_input']['token_type_ids']

        # * forward & backward
        optimizer.zero_grad()

        loss, pred_scores_list = model(
                                encoder_input=batched_data['encoder_input'], unmasked_encoder_input=batched_data['unmasked_encoder_input'],
                                input_ids_list=input_ids, input_masks_list=input_masks, token_type_ids_list=token_type_ids,
                                input_labels_list=input_labels)

        # * normalized by n_sentence and batch size
        if loss is not None:
            n_sen = num_sen.detach().sum()
            if is_distributed() and get_world_size() > 1:
                reduced_n_sen = reduce_tensor(n_sen.float())
            else:
                reduced_n_sen = n_sen

            loss /= (reduced_n_sen.float().item()*get_world_size())
            loss /= float(opt.batch_size)
        
        if not torch.isnan(loss):
            loss.backward()

        if opt.grad_clip != -1:  # * enable, -1 == disable
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

        optimizer.step()

        # * update model parameters with ema
        if ema is not None:
            ema(model, niter)

        # * keep logs
        n_correct = 0
        n_word = 0
        for pred, gold in zip(pred_scores_list, input_labels[-1]):
            n_correct += cal_performance(pred, gold)
            valid_label_mask = gold.ne(VARDataset.IGNORE)
            n_word += valid_label_mask.sum()

        n_word_total += n_word 
        n_word_correct += n_correct
        total_loss += loss 

        if batch_idx%50==0:
            dist_log('[Training epoch{}:{}/{}] acc:{:.5f} loss:{:.5f}'.format(\
                epoch, batch_idx, len(training_data), float(n_word_correct/float(n_word_total)), float(total_loss/float(n_word_total))))

    if is_distributed() and get_world_size() > 1:
        reduced_n_word_total = reduce_tensor(n_word_total.float())
        reduced_total_loss = reduce_tensor(total_loss.float())
        reduced_n_word_correct = reduce_tensor(n_word_correct.float())
    else:
        reduced_n_word_total = n_word_total
        reduced_total_loss = total_loss
        reduced_n_word_correct = n_word_correct

    loss_per_word = 1.0 * reduced_total_loss / reduced_n_word_total
    accuracy = 1.0 * reduced_n_word_correct / reduced_n_word_total

    return float(loss_per_word), float(accuracy)

