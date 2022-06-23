""" This module will handle the text generation with beam search. """

import torch
import torch.nn.functional as F

from src.model.mainmodel import Reasoner
from src.dataloader import VARDataset
from src.utils import get_rank, is_distributed, reduce_tensor, get_world_size

import logging
logger = logging.getLogger(__name__)


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()

    return x


def mask_tokens_after_eos(input_ids, input_masks,
                          eos_token_id=VARDataset.EOS, pad_token_id=VARDataset.PAD):
    """replace values after `[EOS]` with `[PAD]`"""
    for row_idx in range(len(input_ids)):
        # possibly more than one `[EOS]`
        cur_eos_idxs = (input_ids[row_idx] == eos_token_id).nonzero()
        if len(cur_eos_idxs) != 0:
            cur_eos_idx = cur_eos_idxs[0, 0].item()
            input_ids[row_idx, cur_eos_idx+1:] = pad_token_id
            input_masks[row_idx, cur_eos_idx+1:] = 0
    return input_ids, input_masks


class Translator(object):
    """Load with trained model and handle the greedy search"""
    def __init__(self, opt, checkpoint, model=None):
        self.opt = opt
        self.device = torch.device("cuda" if opt.cuda else "cpu")
        if checkpoint:
            self.model_config = checkpoint["model_cfg"]
            self.max_t_len = self.model_config.max_t_len
            self.max_v_len = self.model_config.max_v_len
            self.max_n_len = self.model_config.max_n_len
            self.num_hidden_layers = self.model_config.num_hidden_layers
        else: # mainly used for bug checking
            self.max_t_len = 22
            self.max_v_len = 50
            self.max_n_len = 12

        if model is None:
            model = Reasoner(self.model_config).to(self.device)
            model.load_state_dict(checkpoint["model"])

        if get_rank() <= 0:
            logging.info("[Info] Trained model state loaded.")
            
        self.model = model
        self.model.eval()


    def _check_masked(self, _input_ids_lists):
        for input_ids_list in _input_ids_lists:
            for cur_input_masks in input_ids_list:
                assert torch.sum(cur_input_masks[:, -(self.max_t_len):]) == 0, \
                    "Initially, all text tokens should be masked"


    def greedy_decoding_step(self, k, query_clip, input_ids, video_embeddings, input_masks, token_type_ids,
                            model, sentence_embedding=None, start_idx=VARDataset.BOS, unk_idx=VARDataset.UNK,
                            confidence_vector=None
                            ):

        if sentence_embedding is None:
            dec_sta=self.max_v_len
            dec_end=self.max_v_len + self.max_t_len
        else:
            assert k >= 1
            dec_sta=self.max_v_len + self.max_n_len
            dec_end=self.max_v_len + self.max_n_len + self.max_t_len

        embeddings_layer=model.embeddings_decs[k]
        decoder_layer=model.decoders[k]
        prediction_head=model.pred_heads[k]

        bsz = len(input_ids)
        next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )
        for dec_idx in range(dec_sta, dec_end):
            input_ids[:, dec_idx] = next_symbols
            input_masks[:, dec_idx] = 1

            pred_scores = model.forward_for_translate(
                            video_embeddings=video_embeddings, 
                            input_ids=input_ids, 
                            token_type_ids=token_type_ids, 
                            input_masks=input_masks,
                            query_clip=query_clip,
                            embeddings_layer=embeddings_layer, 
                            decoder_layer=decoder_layer, 
                            prediction_head=prediction_head,
                            sentence_embedding=sentence_embedding,
                            confidence_vector=confidence_vector
                            )
            # suppress unk token; (N, L, vocab_size)
            pred_scores[:, :, unk_idx] = -1e10
            next_words = pred_scores[:, dec_idx].max(1)[1] 
            next_symbols = next_words

        input_ids, input_masks = mask_tokens_after_eos(input_ids, input_masks)

        return input_ids, input_masks, input_ids[:, -(self.max_t_len):]


    def translate_batch_greedy(self, 
                            video_features, 
                            input_ids_list, input_masks_list, token_type_ids_list,
                            temporal_tokens=None, rt_model=None, video_mask=None
                            ):
        checked_list = []
        for k in range(self.opt.K):
            input_ids_list[k], input_masks_list[k] = self.prepare_video_only_inputs(
                input_ids_list[k], input_masks_list[k], token_type_ids_list[k], token_type_id=VARDataset.TEX_TYPE)
            checked_list.append(input_ids_list[k])

        self._check_masked(checked_list)
    
        with torch.no_grad():
            video_embeddings = rt_model.video_encoder(video_features=video_features, video_mask=video_mask, temporal_tokens=temporal_tokens)
            vids_size = len(input_ids_list[0])

            _translated_input_ids_list_dec1 = []
            _translated_input_masks_list_dec1 = []

            _dec1_seq_list = []
            # * first translate for dec1
            for idx in range(vids_size):
                _decoded_id, _decoded_mask, dec_seq = self.greedy_decoding_step(
                    0,
                    idx,
                    input_ids_list[0][idx], video_embeddings,
                    input_masks_list[0][idx], token_type_ids_list[0][idx],
                    model=rt_model
                    )

                _translated_input_ids_list_dec1.append(_decoded_id)
                _translated_input_masks_list_dec1.append(_decoded_mask)
                _dec1_seq_list.append(dec_seq)

            current_seq_list = None
            prev_translated_input_ids_list = _translated_input_ids_list_dec1
            prev_translated_input_masks_list = _translated_input_masks_list_dec1
            prev_sentence_embeddings = None
            prev_confidence_vector = None
            prev_video_embeddings = video_embeddings

            for k in range(1, self.opt.K):
                current_translated_input_ids_list = []
                current_translated_input_masks_list = []

                sentence_embedding, prev_outputs, confidence_vector = \
                                        rt_model.construct_sentence_emb_for_translate(
                                                prev_translated_input_ids_list,
                                                prev_translated_input_masks_list,
                                                token_type_ids_list[k-1],
                                                prev_video_embeddings,
                                                prev_confidence_vector=prev_confidence_vector,
                                                prev_sentence_embeddings=prev_sentence_embeddings,
                                                embeddings_layer=rt_model.embeddings_decs[k-1],  
                                                decoder_layer=rt_model.decoders[k-1],  
                                                pred_head=rt_model.pred_heads[k-1] 
                                            )

                S = len(input_ids_list[0])
                SB,_,D = prev_outputs.shape
                B = int(SB/S)
                prev_outputs = prev_outputs[:, :(self.opt.max_v_len), :].view(S, B, self.opt.max_v_len, D)
                current_seq_list = []
                prev_sentence_embeddings = []
                prev_confidence_vector = []
                # * dec2 for final predict
                for idx in range(vids_size):
                    _video_embeddings = prev_outputs[idx]

                    # * confidence_vector S*B, N
                    _confidence_vector = confidence_vector.view(S,B,*confidence_vector.shape[1:])[idx]
                    
                    _sentence_embedding = sentence_embedding

                    prev_sentence_embeddings.append(_sentence_embedding)
                    prev_confidence_vector.append(_confidence_vector)

                    _decoded_id, _decoded_mask, dec_seq = self.greedy_decoding_step(
                        k,
                        idx,
                        input_ids_list[k][idx], _video_embeddings,
                        input_masks_list[k][idx], token_type_ids_list[k][idx],
                        model=rt_model,
                        sentence_embedding=_sentence_embedding,
                        confidence_vector=_confidence_vector
                        )

                    current_translated_input_ids_list.append(_decoded_id)
                    current_translated_input_masks_list.append(_decoded_mask)
                    current_seq_list.append(dec_seq)

                prev_sentence_embeddings = torch.stack(prev_sentence_embeddings,dim=0).contiguous().view(-1,*_sentence_embedding.shape[1:])
                prev_confidence_vector = torch.stack(prev_confidence_vector,dim=0).contiguous().view(-1,*_confidence_vector.shape[1:])
                prev_video_embeddings = prev_outputs.view(-1, self.opt.max_v_len, D)
                prev_translated_input_ids_list = current_translated_input_ids_list
                prev_translated_input_masks_list = current_translated_input_masks_list
                    
            return current_seq_list if current_seq_list is not None else _dec1_seq_list
            

    def translate_batch(self, batched_data):

        num_sen = batched_data['num_sen']
        video_features = batched_data['encoder_input']["video_features"]
        temporal_tokens = batched_data['encoder_input']["temporal_tokens"]
        video_mask = batched_data['encoder_input']["video_mask"]
        # * decoders
        input_labels = batched_data["gt"]
        input_ids = batched_data['decoder_input']["input_ids"]
        input_masks = batched_data['decoder_input']["input_mask"]
        token_type_ids = batched_data['decoder_input']["token_type_ids"]

        return self.translate_batch_greedy(
            input_ids_list=input_ids, input_masks_list=input_masks, token_type_ids_list=token_type_ids,
            rt_model=self.model, 
            temporal_tokens=temporal_tokens, video_mask=video_mask, video_features=video_features)


    @classmethod
    def prepare_video_only_inputs(cls, input_ids, input_masks, segment_ids, token_type_id=1):
        """ 
        replace text_ids (except `[BOS]`) in input_ids with `[PAD]` token, for decoding.
        """
        video_only_input_ids_list = []
        video_only_input_masks_list = []
        for e1, e2, e3 in zip(input_ids, input_masks, segment_ids):
            text_mask = (e3 == token_type_id)  # text positions (`1`) are replaced
            e1[text_mask] = VARDataset.PAD
            e2[text_mask] = 0  # mark as invalid bits
            video_only_input_ids_list.append(e1)
            video_only_input_masks_list.append(e2)

        return video_only_input_ids_list, video_only_input_masks_list