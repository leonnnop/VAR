import sys
import json
import copy
import math
import token

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from easydict import EasyDict as edict

from .utils.optimization import LabelSmoothingLoss
from .modules import BertLayerNorm, BertLMPredictionHead
from .visual_model import VideoEncodingTrans
from .linguistic_model import TextEmbedding, BertEmbeddingsWithVideo, BertEmbeddingsWithSentence, BertEncoder, BertEncoderSen
from ..utils import dist_log

import logging
logger = logging.getLogger(__name__)


def _concat_list_of_tensors(_list):
    outs = []
    for param in _list:
        out = torch.stack(param, dim=0).contiguous().view(-1, *param[0].shape[1:])
        outs.append(out)
    return outs

class Reasoner(nn.Module):
    def __init__(self, config):
        super(Reasoner, self).__init__()
        self.config = config

        self.video_encoder = VideoEncodingTrans(config, add_postion_embeddings=True)
        self.projection_head_q = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(), nn.Linear(config.hidden_size, config.hidden_size))

        self.video_encoder_aux = VideoEncodingTrans(config, add_postion_embeddings=True)
        self.projection_head_k = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(), nn.Linear(config.hidden_size, config.hidden_size))
        self.m = self.config.momentum_aux_m

        for q, k in zip((self.video_encoder.parameters(), self.projection_head_q.parameters()), (self.video_encoder_aux.parameters(), self.projection_head_k.parameters())):
            for param_q, param_k in zip(q, k):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient 

        if self.config.K > 1:
            shared_embeddings_dec = BertEmbeddingsWithSentence(config, add_postion_embeddings=True)
            shared_decoder = BertEncoderSen(config)
            shared_pred_head = BertLMPredictionHead(config, None)

            self.embeddings_decs = nn.ModuleList([
                                    BertEmbeddingsWithVideo(config, add_postion_embeddings=True) if k==0 else \
                                    shared_embeddings_dec \
                                    for k in range(self.config.K)
                                    ])
            self.decoders = nn.ModuleList([BertEncoder(config) if k==0 else shared_decoder for k in range(self.config.K)])
            self.pred_heads = nn.ModuleList([BertLMPredictionHead(config, None) if k==0 else shared_pred_head for k in range(self.config.K)])
        else:
            # * in list form just for compatibility
            self.embeddings_decs = nn.ModuleList([BertEmbeddingsWithVideo(config, add_postion_embeddings=True)])
            self.decoders = nn.ModuleList([BertEncoder(config)])
            self.pred_heads = nn.ModuleList([BertLMPredictionHead(config, None)])

        self.loss_func = LabelSmoothingLoss(config.label_smoothing, config.vocab_size, ignore_index=-1) \
            if "label_smoothing" in config and config.label_smoothing > 0 else nn.CrossEntropyLoss(ignore_index=-1)
        self.apply(self.init_bert_weights)

        # * scheduled sampling
        self.probs = 0.

        # * confidence embedding
        self.conf_bucket_size = config.conf_bucket_size

        # * sentence embedding
        if self.config.sentence_emb_aggregation_mode == 'weighted':
            self._weighted_para = nn.Sequential(
                                    nn.Linear(config.hidden_size, 4*config.hidden_size),
                                    nn.GELU(),
                                    nn.Linear(4*config.hidden_size, 1),
                                )


    def init_bert_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    # * scheduled sampling
    def _update_scheduled_sampling(self, _pred, input_ids):
        _pred = torch.cat([(torch.zeros((int(_pred.shape[0]), 1, int(_pred.shape[2])))).to(_pred.device).float(), _pred], dim=1)[:, :-1]
        # * pred_size S*B,L,vocab_size -> S*B,L
        pred = _pred.max(2)[1]
        # * for each word, p prob to be replaced by predicted results
        # * S*B,L
        prob = torch.softmax(_pred, dim=2).max(2)[0]
        replace_mask = (prob > self.config.conf_replace_tr)
        # * DO NOT replace video ids + [BOS]
        replace_mask[:,:(self.config.max_v_len+1)] = False

        input_ids[replace_mask] = pred[replace_mask]

        return input_ids

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.decay_prob(epoch, k=self.config.decay_k)

    def decay_prob(self, i, or_type=3, k=3000):
        if or_type == 1:  # Linear decay
            or_prob_begin, or_prob_end = 1., 0.
            or_decay_rate = (or_prob_begin - or_prob_end) / 10.
            ss_decay_rate = 0.1
            prob = or_prob_begin - (ss_decay_rate * i)
            if prob < or_prob_end:
                prob_i = or_prob_end
                dist_log('[Linear] schedule sampling probability do not change {}'.format(prob_i))
            else:
                prob_i = prob
                dist_log('[Linear] decay schedule sampling probability to {}'.format(prob_i))

        elif or_type == 2:  # Exponential decay
            prob_i = np.power(k, i)
            dist_log('[Exponential] decay schedule sampling probability to {}'.format(prob_i))

        elif or_type == 3:  # Inverse sigmoid decay
            prob_i = k / (k + np.exp((i / k)))
            dist_log('[Inverse] decay schedule sampling probability to {}'.format(prob_i))
        self.probs = prob_i

        return prob_i

    def get_word_orcale_tokens(self, _pred, prev_output_tokens, epsilon=1e-6):
        _gumbel_noise = 0.5 

        B, L = prev_output_tokens.size()
        pred_logits = _pred[:, self.config.max_v_len:]
        # B x L x V
        pred_logits.add_(-torch.log(-torch.log(torch.Tensor(
            pred_logits.size()).to(pred_logits.device).uniform_(0, 1) + epsilon) + epsilon)) / _gumbel_noise

        pred_tokens = torch.max(pred_logits, dim=-1)[1]
        bos_idx = prev_output_tokens[0, self.config.max_v_len]
        pred_tokens = torch.cat([(bos_idx * torch.ones((B, 1)).to(pred_tokens.device)), pred_tokens], dim=1)[:, :-1]

        sample_gold_prob = self.probs 
        sample_gold_prob = sample_gold_prob * torch.ones_like(prev_output_tokens[:, self.config.max_v_len:], dtype=torch.float32)
        sample_gold_mask = torch.bernoulli(sample_gold_prob).long()

        updated_tokens = prev_output_tokens[:, self.config.max_v_len:] * sample_gold_mask + pred_tokens * (1 - sample_gold_mask)
        prev_output_tokens[:, self.config.max_v_len:] = updated_tokens

        return prev_output_tokens

    def _manage_scheduled_sampling(self, prediction_scores_dec1_pass1, input_ids):
        if self.config.scheduled_method == 'confidence':
            input_ids = self._update_scheduled_sampling(prediction_scores_dec1_pass1.detach(), input_ids)
        elif self.config.scheduled_method == 'probability':
            input_ids = self.get_word_orcale_tokens(prediction_scores_dec1_pass1.detach(), input_ids)
        else:
            raise ValueError("Unsupported Method {}".format(self.config.scheduled_method))

        return input_ids

    # * forward (train or translate)
    def _dec1_pass(self, input_ids, input_masks, token_type_ids, video_embeddings):
        embeddings_dec1 = self.embeddings_decs[0](input_ids, token_type_ids, video_embeddings=video_embeddings)  # (N, L, D)

        dec1_outputs = self.decoders[0](
            embeddings_dec1, input_masks, output_all_encoded_layers=False)[-1]  # both outputs are list
        prediction_scores_dec1 = self.pred_heads[0](dec1_outputs)  # (S*B, L, vocab_size) 

        return prediction_scores_dec1, dec1_outputs

    def forward_for_translate(self, query_clip, video_embeddings, input_ids, token_type_ids, input_masks,
                              sentence_embedding=None,
                              embeddings_layer=None, decoder_layer=None, prediction_head=None,
                              confidence_vector=None
                              ):
        embeddings = embeddings_layer(input_ids, token_type_ids, query_clip=query_clip, video_embeddings=video_embeddings, sentence_embeddings=sentence_embedding)  # (N, L, D)
        encoded_layer_outputs = decoder_layer(
            embeddings, input_masks, output_all_encoded_layers=False, confidence_vector=confidence_vector)  # both outputs are list
        prediction_scores = prediction_head(encoded_layer_outputs[-1])  # (N, L, vocab_size)

        return prediction_scores

    # * sentence embeddings
    def _meta_sen_emb_construction(self, metas, input_masks, dec1_outputs):
        B,S,D = metas

        sentence_embeddings = (input_masks.int().unsqueeze(-1) * dec1_outputs)[:, -(self.config.max_t_len):, :]
        
        if self.config.sentence_emb_aggregation_mode == 'mean':
            sentence_embeddings = torch.div(torch.sum(sentence_embeddings, dim=1), 
                        torch.sum(input_masks.int()[:, -(self.config.max_t_len):], dim=1).unsqueeze(-1))
        elif self.config.sentence_emb_aggregation_mode == 'max':
            sentence_embeddings = torch.max(sentence_embeddings, dim=1).values
        elif self.config.sentence_emb_aggregation_mode == 'weighted':
            _weighted = torch.softmax(self._weighted_para(sentence_embeddings), dim=1)
            sentence_embeddings = torch.bmm(_weighted.transpose(1,2), sentence_embeddings)
            sentence_embeddings = sentence_embeddings.squeeze(1)
        else:
            raise ValueError("Unsupported Aggregation Mode {}".format(self.config.sentence_emb_aggregation_mode))

        sentence_embeddings = sentence_embeddings.view(S,B,D).transpose(0,1)

        return sentence_embeddings

    def construct_sentence_emb_for_translate(self, input_ids_list_prev, input_masks_list_prev, token_type_ids_list_prev, video_embeddings, 
    embeddings_layer, decoder_layer, pred_head, prev_sentence_embeddings=None, prev_confidence_vector=None):
        # * dec1
        input_ids, token_type_ids, input_masks = \
                            _concat_list_of_tensors([input_ids_list_prev, token_type_ids_list_prev, input_masks_list_prev])

        embeddings_prev = embeddings_layer(input_ids, token_type_ids, video_embeddings=video_embeddings, sentence_embeddings=prev_sentence_embeddings)  # (N, L, D)

        prev_outputs = decoder_layer(
            embeddings_prev, input_masks, confidence_vector=prev_confidence_vector, output_all_encoded_layers=False)[-1]  # both outputs are list
        prediction_scores_prev = pred_head(prev_outputs)  # (S*B, L, vocab_size) 

        B, S, D = input_ids_list_prev[0].shape[0], len(input_ids_list_prev), prev_outputs.shape[-1] 
        metas = (B,S,D) 

        sentence_embeddings, confidence_vector = self._conf_n_sen_emb_construction(metas, prediction_scores_prev, prev_outputs, input_masks)

        return sentence_embeddings, prev_outputs, confidence_vector


    def _conf_n_sen_emb_construction(self, metas, prediction_scores_dec1, dec1_outputs, input_masks, n_S=1):
        B,S,D = metas

        _sentence_embeddings = self._meta_sen_emb_construction(metas, input_masks, dec1_outputs)
        sentence_embeddings = _sentence_embeddings.repeat(n_S,1,1)

        confidence_vector = self._meta_conf_vec_construction(metas, prediction_scores_dec1, _sentence_embeddings, input_masks)

        confidence_vector = confidence_vector.repeat(n_S,1)
        try:
            assert confidence_vector.max() < self.config.conf_bucket_size
            assert confidence_vector.min() >= 0
        except:
            confidence_vector = torch.clamp(confidence_vector, min=0, max=(self.config.conf_bucket_size-1))

        N = (self.config.max_v_len+self.config.max_n_len+self.config.max_t_len)
        _sentence_embeddings = torch.zeros(n_S*B, N, self.config.hidden_size, dtype=sentence_embeddings.dtype, device=sentence_embeddings.device)
        _sentence_embeddings[:, self.config.max_v_len:self.config.max_v_len+self.config.max_n_len] = sentence_embeddings

        N = (self.config.max_v_len+self.config.max_n_len+self.config.max_t_len)
        assert confidence_vector.shape[1] == S
        # whatever in training or inference, return S*B. and would be clipped later
        if confidence_vector.shape[0] != (S*B):
            confidence_vector = confidence_vector.repeat(S,1)

        _confidence_vector = confidence_vector.view(S,B,S)[0].view(B,S,1) # * B,S,1
        _confidence_vector = _confidence_vector.repeat(1,1,N)
        _confidence_vector = _confidence_vector.transpose(0,1)
        _confidence_vector = _confidence_vector.reshape(-1,N) # * S,B,N -> S*B,N

        _confidence_vector[:, self.config.max_v_len:self.config.max_v_len+self.config.max_n_len] = confidence_vector

        assert _confidence_vector.max() < self.config.conf_bucket_size
        assert _confidence_vector.min() >= 0 

        return _sentence_embeddings, _confidence_vector


    # * confidence embeddings
    @torch.no_grad()
    def _meta_conf_vec_construction(self, metas, prediction_scores_dec1, senten_emb, input_masks):
        confidence_vector = self._generate_conf_matrix_pred(metas, prediction_scores_dec1, input_masks) # * B, S 
        return confidence_vector.detach()

    @torch.no_grad()
    def _generate_conf_matrix_pred(self, metas, pred, input_masks):
        temprature = self.config.conf_temperature
        B,S,_ = metas
        # * mask: S*B,N
        # * pred: S*B, N, voab
        _word_score = torch.max(pred, dim=-1).values # * S*B, N
        _word_score = (input_masks.int() * _word_score)[:, -(self.config.max_t_len):] # * S*B, L

        _sen_score = torch.div(torch.sum(_word_score, dim=1), 
                        torch.sum(input_masks.int()[:, -(self.config.max_t_len):], dim=1)) # * S*B, 
        _sen_score = _sen_score.view(S,B).transpose(0,1) # * B, S 
        _sen_score = torch.softmax(_sen_score/temprature, dim=1)

        # * quantize
        _sen_score = (_sen_score*self.conf_bucket_size).floor().long()

        return _sen_score

    @torch.no_grad()
    def _momentum_update_aux_encoder(self):
        """
        Momentum update of the aux encoder
        """
        for param_q, param_k in zip(self.video_encoder.parameters(), self.video_encoder_aux.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.projection_head_q.parameters(), self.projection_head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
    
    def _forward_refactored(self, 
                    encoder_input, unmasked_encoder_input, 
                    input_ids_list, input_masks_list, token_type_ids_list,
                    input_labels_list=None,
                    ):
        # * forward while training or validation

        caption_loss = 0.
        prediction_scores_list = []

        # * masked forward
        video_embeddings = self.video_encoder(**encoder_input)

        # aux video loss
        _video_embeddings = self.projection_head_q(video_embeddings)
        # * unmasked forward
        with torch.no_grad():
            self._momentum_update_aux_encoder()  # update the key encoder

            unmasked_video_embeddings = self.video_encoder_aux(**unmasked_encoder_input)
            _unmasked_video_embeddings = self.projection_head_k(unmasked_video_embeddings)

            _unmasked_video_embeddings = _unmasked_video_embeddings * unmasked_encoder_input['video_mask'][...,None]
            _unmasked_video_embeddings = _unmasked_video_embeddings.detach_()

        # * only unmasked part need to calculate loss
        _video_embeddings = _video_embeddings * unmasked_encoder_input['video_mask'][...,None]

        aux_loss = self.config.loss_aux_weight * F.mse_loss(_video_embeddings, _unmasked_video_embeddings, reduction='sum')
        aux_loss = aux_loss / torch.sum(unmasked_encoder_input['video_mask']).detach_()
        caption_loss += aux_loss

        B = video_embeddings.shape[0]
        # * dec1
        input_labels, input_ids, token_type_ids, input_masks = \
            _concat_list_of_tensors([input_labels_list[0], input_ids_list[0], token_type_ids_list[0], input_masks_list[0]])

        if not self.config.disable_scheduled_sampling:
            with torch.no_grad():
                prediction_scores_dec1_pass1, _ = self._dec1_pass(input_ids, input_masks, token_type_ids, video_embeddings)
                input_ids = self._manage_scheduled_sampling(prediction_scores_dec1_pass1, input_ids)

        prediction_scores_dec1, dec1_outputs = self._dec1_pass(input_ids, input_masks, token_type_ids, video_embeddings)

        B, S, D = video_embeddings.shape[0], len(input_ids_list[0]), dec1_outputs.shape[-1] 
        metas = (B,S,D)

        # * iterative decoder
        prev_outputs = dec1_outputs
        prev_prediction_scores = prediction_scores_dec1
        for k in range(1, self.config.K):
            caption_loss += (self.config.loss_aux_caption)*self.loss_func(prev_prediction_scores.view(-1, self.config.vocab_size),
                                    input_labels.view(-1))
                                    
            # * sentence embeddings and confidence embeddings
            sentence_embeddings, confidence_vector = self._conf_n_sen_emb_construction(metas, prev_prediction_scores, prev_outputs, input_masks, n_S=S)

            # * dec2
            input_labels, input_ids, token_type_ids, input_masks = \
                _concat_list_of_tensors([input_labels_list[k], input_ids_list[k], token_type_ids_list[k], input_masks_list[k]])

            # * update input ids with previous prediction
            prev_pred_id = prev_prediction_scores.max(2).values # S*B, L
            input_ids[:,-self.config.max_t_len] = prev_pred_id[:,-self.config.max_t_len].detach()

            _video_embeddings = prev_outputs[:, :(self.config.max_v_len), :]

            embeddings_dec2 = self.embeddings_decs[k](input_ids, token_type_ids, 
                                                        video_embeddings=_video_embeddings,
                                                        sentence_embeddings=sentence_embeddings
                                                        )  # (N, L, D)

            current_outputs = self.decoders[k](
                embeddings_dec2, input_masks, output_all_encoded_layers=False, confidence_vector=confidence_vector)[-1]  # both outputs are list

            # * final pred
            current_prediction_scores = self.pred_heads[k](current_outputs)  # (S*B, L, vocab_size)
            prediction_scores_list = current_prediction_scores.view(S,B,*current_prediction_scores.shape[1:])

            # * for next decoder
            prev_outputs = current_outputs
            prev_prediction_scores = current_prediction_scores

        # * final calculate loss
        if self.config.K == 1:
            current_prediction_scores = prev_prediction_scores
            prediction_scores_list = current_prediction_scores.view(S,B,*current_prediction_scores.shape[1:])
        caption_loss += self.config.loss_main_caption*self.loss_func(current_prediction_scores.view(-1, self.config.vocab_size),
                                    input_labels.view(-1))
        

        return caption_loss, prediction_scores_list


    def forward(self, **kwargs):
        return self._forward_refactored(**kwargs)
