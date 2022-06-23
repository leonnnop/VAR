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
from .modules import BertLayerNorm, PositionEncoding, BertAttention, BertIntermediate, BertOutput


import logging
logger = logging.getLogger(__name__)

class BertLayerUntied(nn.Module):
    def __init__(self, config):
        super(BertLayerUntied, self).__init__()
        config.enable_relative = True

        self.config = config
        self.attention = BertAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, diagonal_mask=False, temporal_tokens=None):

        self_attention_mask = attention_mask.unsqueeze(1)
        if diagonal_mask:  # mask subsequent words
            max_len = hidden_states.size(1)
            self_attention_mask = self_attention_mask * \
                torch.tril(self_attention_mask.new_ones(max_len, max_len), diagonal=0)
        attention_output = self.attention(hidden_states, self_attention_mask, temporal_tokens=temporal_tokens)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D)
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        return layer_output


class BertEncoderUntied(nn.Module):
    def __init__(self, config):
        super(BertEncoderUntied, self).__init__()
        self.layer = nn.ModuleList([BertLayerUntied(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, 
                diagonal_mask=False, output_all_encoded_layers=False, temporal_tokens=None):

        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask, diagonal_mask, 
                            temporal_tokens=temporal_tokens)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        
        return all_encoder_layers


class VideoEncodingTrans(nn.Module):

    def __init__(self, config, add_postion_embeddings=True):
        super(VideoEncodingTrans, self).__init__()

        self.video_embeddings = nn.Sequential(
            BertLayerNorm(config.video_feature_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.video_feature_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        self.add_postion_embeddings = add_postion_embeddings
        
        if self.add_postion_embeddings:
            self.position_embeddings = PositionEncoding(n_filters=config.hidden_size,
                                                        max_len=(config.max_v_len * config.max_n_len))
        self.encoder = BertEncoderUntied(config)

        self.config = config


    def forward(self, video_features, video_mask, temporal_tokens):
        embeddings = self.video_embeddings(video_features)

        B,N,D = embeddings.shape
        # B,N,D
        if self.add_postion_embeddings:
            embeddings = self.position_embeddings(embeddings)

        video_features = self.encoder(embeddings, video_mask, temporal_tokens=temporal_tokens)[-1]

        return video_features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


