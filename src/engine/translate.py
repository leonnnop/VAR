""" Translate input text with trained model. """

import os
from collections import defaultdict
from easydict import EasyDict as EDict

import torch
from src.dataloader import prepare_batch_inputs
from src.engine.translator import Translator
from src.utils import save_json, load_json, get_rank, is_distributed, get_world_size, dist_log

from eval_kit.evaluate2021 import main as eval_dvc

def remove_duplicate(res_dict):
    ret_res_dict = {}
    for k, v in res_dict.items():
        seen = set()
        new_v = []
        for d in v:
            t = tuple(d.items())
            if t not in seen:
                seen.add(t)
                new_v.append(d)
        ret_res_dict[k] = new_v
        
    return ret_res_dict

def sort_res(res_dict):
    res_dict = remove_duplicate(res_dict)
    final_res_dict = {}
    for k, v in res_dict.items():
        final_res_dict[k] = sorted(v, key=lambda x: float(x["clip_idx"]))
        
    return final_res_dict

def merge_final_results(file_list):
    batch_res_all = [load_json(e) for e in file_list]
    batch_res = batch_res_all[0]
    for res in batch_res_all[1:]:
        for name in res.keys():
            if name in batch_res:
                batch_res[name].extend(res[name])
            else:
                batch_res[name] = res[name]
        
    batch_res = sort_res(batch_res)

    return batch_res

def run_translate(eval_data_loader, translator, opt):
    # submission template
    batch_res = defaultdict(list)
                 
    for batch_idx, raw_batch in enumerate(eval_data_loader):
        if batch_idx%5==0:
            dist_log('[Translate {}/{}]'.format(batch_idx, len(eval_data_loader)))
        # * meta: [b0[v0,v1,v2], b1[], b2[]]
        meta_all = raw_batch[1] 

        batched_data = prepare_batch_inputs(raw_batch[0], device=translator.device)

        # * dec_seq_list: [v0[b0,b1...],v1[],[]]
        dec_seq_list = translator.translate_batch(batched_data)

        for _v in range(len(dec_seq_list)):
            dec_seq = dec_seq_list[_v]   
            meta = [b[_v] if _v < len(b) else None for b in meta_all]
            
            # * example_idx indicates which example it is in the batch
            for example_idx, (cur_gen_sen, cur_meta) in enumerate(zip(dec_seq, meta)):
                if cur_meta is None:
                    continue

                cur_data = {
                    "sentence": eval_data_loader.dataset.convert_ids_to_sentence(
                        cur_gen_sen.cpu().tolist()),
                    "gt_sentence": cur_meta["gt_sentence"],
                    "clip_idx": cur_meta["clip_idx"], # used for sort
                    "is_hypothesis": cur_meta["is_hypothesis"],
                }
                
                batch_res[cur_meta["name"]].append(cur_data)

    batch_res = sort_res(batch_res)

    return batch_res

def translate_w_metrics(checkpoint, eval_data_loader, opt, model=None, eval_mode='test'):
    all_metrics, res_filepath, merged_res_filepath, all_metrics_filepath = None, None, None, None

    with torch.no_grad():
        translator = Translator(opt, checkpoint, model=model)
        json_res = run_translate(eval_data_loader, translator, opt=opt)

        res_filepath = os.path.abspath(opt.save_model + '_tmp_greedy_pred_{}_{}.json'.format(eval_mode, int(get_rank())))
        save_json(json_res, res_filepath, save_pretty=True)

        # * wait all progress to save json file
        if is_distributed() and get_world_size() > 1:
            torch.distributed.barrier()

    if opt.local_rank <= 0:
        dist_log('Start evaluating with [{}] mode.'.format(opt.score_mode))

        # * first combine files produced from all process
        file_list = [os.path.abspath(opt.save_model + '_tmp_greedy_pred_{}_{}.json'.format(eval_mode, rank)) for rank in range(get_world_size())]
        merged_json_res = merge_final_results(file_list)
        merged_res_filepath = os.path.abspath(opt.save_model + '_tmp_greedy_pred_{}.json'.format(eval_mode))
        save_json(merged_json_res, merged_res_filepath, save_pretty=True)

        # * per event evaluation
        all_metrics = eval_dvc(EDict(submission=merged_res_filepath, separate=False, verbose=False))
        dist_log('Finished eval {}.'.format(eval_mode))

        all_metrics_filepath = merged_res_filepath.replace('.json', '_all_metrics.json')
        save_json(all_metrics, all_metrics_filepath, save_pretty=True)

    if is_distributed() and get_world_size() > 1:
        torch.distributed.barrier()

    return all_metrics, [merged_res_filepath, all_metrics_filepath]
