import sys
import json
import copy
import math
import token

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
import numpy as np
from .modules import BertLayerNorm, PositionEncoding, BertAttention, BertIntermediate, BertOutput, BertSelfAttentionSen
from .utils.helpers import make_pad_shifted_mask

import logging
logger = logging.getLogger(__name__)


class BertLayer(nn.Module):
    def __init__(self, config, sentence_len=0):
        super(BertLayer, self).__init__()
        config.enable_relative = False

        self.config = config
        self.attention = BertAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.sentence_len = sentence_len

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:

        """
        max_v_len, max_t_len, max_n_len = self.config.max_v_len, self.config.max_t_len, self.config.max_n_len
        vid_len = max_v_len

        # self-attention, need to shift right
        shifted_self_mask = make_pad_shifted_mask(attention_mask, vid_len, max_t_len, sentence_len=self.sentence_len)  # (N, L, L)
        attention_output = self.attention(hidden_states, shifted_self_mask)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D)
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        return layer_output

class BertLayerSen(nn.Module):
    def __init__(self, config, sentence_len=0):
        super(BertLayerSen, self).__init__()
        config.enable_relative = True

        self.config = config
        self.attention = BertAttention(config, _func_attn=BertSelfAttentionSen)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.sentence_len = sentence_len

    def forward(self, hidden_states, attention_mask, confidence_vector=None):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:

        """
        max_v_len, max_t_len, max_n_len = self.config.max_v_len, self.config.max_t_len, self.config.max_n_len
        vid_len = max_v_len

        # self-attention, need to shift right
        shifted_self_mask = make_pad_shifted_mask(attention_mask, vid_len, max_t_len, sentence_len=self.sentence_len)  # (N, L, L)
        attention_output = self.attention(hidden_states, shifted_self_mask, confidence_vector=confidence_vector)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D)
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_dec1_blocks)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, confidence_vector=None):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:

        Returns:

        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertEncoderSen(nn.Module):
    def __init__(self, config):
        super(BertEncoderSen, self).__init__()
        self.layer = nn.ModuleList([BertLayerSen(config, sentence_len=config.max_n_len) for _ in range(config.num_dec2_blocks)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, confidence_vector=None):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:

        Returns:

        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask, confidence_vector=confidence_vector)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers



class TextEmbedding(nn.Module):
    def __init__(self, config):
        super(TextEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.word_vec_size, padding_idx=0)
        self.word_fc = nn.Sequential(
            BertLayerNorm(config.word_vec_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.word_vec_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

    def set_pretrained_embedding(self, pretrained_embedding, freeze=True):
        """Note the from_pretrained does not work in-place, so you need to assign value to the embedding"""
        assert pretrained_embedding.shape == self.word_embeddings.weight.shape  # ensure equal shape
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze,
                                                            padding_idx=self.word_embeddings.padding_idx)
    
    def forward(self, input_ids):
        return self.word_fc(self.word_embeddings(input_ids))



class BertEmbeddingsWithVideo(nn.Module):
    def __init__(self, config, add_postion_embeddings=True):
        super(BertEmbeddingsWithVideo, self).__init__()
        """add_postion_embeddings: whether to add absolute positional embeddings"""
        self.add_postion_embeddings = add_postion_embeddings
        
        self.txt_emb = TextEmbedding(config)
        
        if self.add_postion_embeddings:
            self.position_embeddings = PositionEncoding(n_filters=config.hidden_size,
                                                        max_len=(config.max_t_len+config.max_v_len))
        # video as 0, text as 1 -> 2
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.config = config

    def forward(self, input_ids, token_type_ids, video_embeddings, sentence_embeddings=None, query_clip=-1):
        """
        Args:
            input_ids: (N, L)
            video_features: (N, L, D)
            token_type_ids: (N, L, D)

        Returns:

        """

        words_embeddings = self.txt_emb(input_ids)
        
        if query_clip == -1:
            B,_,D = video_embeddings.shape 
            _video_embeddings = video_embeddings.view(B,-1,self.config.max_v_len,D).transpose(0,1).contiguous().view(-1,self.config.max_v_len,D)
        else:
            _video_embeddings = video_embeddings[:,(query_clip*self.config.max_v_len):((query_clip+1)*self.config.max_v_len)]

        video_embeddings = torch.zeros(_video_embeddings.shape[0], 
                                        _video_embeddings.shape[1]+self.config.max_t_len, 
                                        _video_embeddings.shape[2]).to(_video_embeddings.device)
        video_embeddings[:,:_video_embeddings.shape[1]] = _video_embeddings

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + video_embeddings + token_type_embeddings

        if self.add_postion_embeddings:
            embeddings = self.position_embeddings(embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings  # (N, L, D) 


class BertEmbeddingsWithSentence(nn.Module):
    def __init__(self, config, add_postion_embeddings=True):
        super(BertEmbeddingsWithSentence, self).__init__()
        """add_postion_embeddings: whether to add absolute positional embeddings"""
        self.add_postion_embeddings = add_postion_embeddings

        self.txt_emb = TextEmbedding(config)
        
        if self.add_postion_embeddings:
            self.position_embeddings = PositionEncoding(n_filters=config.hidden_size,
                                                        max_len=(config.max_t_len+
                                                                 config.max_n_len+
                                                                 config.max_v_len))
        # video as 0, text as 1, sentence as 2 -> 3
        self.token_type_embeddings = nn.Embedding(3, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.config = config


    def forward(self, input_ids, token_type_ids, video_embeddings, sentence_embeddings, query_clip=-1):
        """
        Args:
            input_ids: (N, L)
            video_features: (N, L, D)
            token_type_ids: (N, L, D)

        Returns:

        """

        words_embeddings = self.txt_emb(input_ids)
        
        # * direct use encoded visual feature from dec1
        _video_embeddings = video_embeddings

        video_embeddings = torch.zeros(_video_embeddings.shape[0], 
                                        self.config.max_v_len+self.config.max_t_len+self.config.max_n_len, 
                                        _video_embeddings.shape[2]).to(_video_embeddings.device)

        video_embeddings[:,:_video_embeddings.shape[1]] = _video_embeddings

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # * process sentence emb
        _sentence_embeddings = sentence_embeddings

        embeddings = words_embeddings + video_embeddings + token_type_embeddings + _sentence_embeddings

        if self.add_postion_embeddings:
            embeddings = self.position_embeddings(embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings  # (N, L, D) 
