import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from src.utils import get_rank

try:
    from .utils.rpe_ops.rpe_index import RPEIndexFunction
except ImportError:
    RPEIndexFunction = None
    import warnings
    RED_STR = "\033[91m{}\033[00m"
    warnings.warn(RED_STR.format("[WARNING] The module `rpe_ops` is not built. \
For better training performance, please build `rpe_ops`."),)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT"s gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(d_model=6, max_len=10, dropout=0)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x, return_pe=False):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        if return_pe: return x, pe
        x = x + pe
        return x


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class RelativeTemEmbedding(nn.Module):
    def __init__(self, config, in_lin=False,
                    on_value=False, relative_mode='bias', 
                    relative_directional=True, relative_multihead=False, num_buckets=10):
        super(RelativeTemEmbedding, self).__init__()
        # config.num_attention_heads
        self.head_dim = int(config.hidden_size/config.num_attention_heads)
        if relative_directional:
            self.num_buckets = 2*(num_buckets)-1
        else:
            self.num_buckets = num_buckets

        if relative_multihead:
            self.num_heads = config.num_attention_heads
        else:
            self.num_heads = 1

        if relative_mode == 'bias':
            self.lookup_table_bias = nn.Parameter(
                        torch.zeros(self.num_heads, self.num_buckets))
        elif relative_mode == 'contextual':
            if on_value:
                self.lookup_table_weight = nn.Parameter(
                    torch.zeros(self.num_heads, self.num_buckets, self.head_dim))
            else:
                self.lookup_table_weight = nn.Parameter(
                            torch.zeros(self.num_heads, self.head_dim, self.num_buckets))

        self.config = config
        self.on_value = on_value
        self.relative_mode = relative_mode
        self.relative_directional = relative_directional
        self.relative_multihead = relative_multihead
        self.in_lin = in_lin

    def _meta_forward(self, rp_bucket, query_emb):
        # * rp_bucket: L,L
        # * query_emb: B,H,L,D

        L = rp_bucket.shape[0]
        if query_emb is None and self.relative_mode == 'bias':
            
            embs = torch.index_select(self.lookup_table_bias, 1, rp_bucket.flatten()).\
                    view(1, self.num_heads, L, L)

        elif self.relative_mode == 'contextual':
            B = query_emb.shape[0]
            if self.on_value:
                weight = torch.index_select(self.lookup_table_weight, 1, rp_bucket.flatten()).view(1, L, L, self.head_dim)
                # (H, L_query, B, L_key) @ (H, L_query, L_key, D) = (H, L_query, B, D)
                # -> (B, H, L_query, D)
                embs = torch.matmul(query_emb.permute(1, 2, 0, 3), weight).permute(2, 0, 1, 3)
            else:
                lookup_table = torch.matmul(
                    query_emb.transpose(0, 1).reshape(-1, B * L, self.head_dim),
                    self.lookup_table_weight).\
                    view(-1, B, L, self.num_buckets).transpose(0, 1)

                if RPEIndexFunction is not None:
                    rp_bucket = rp_bucket.to(torch.int32)
                    embs = RPEIndexFunction.apply(lookup_table, rp_bucket)
                else:
                    offset = torch.arange(0, L * self.num_buckets, self.num_buckets,
                                            dtype=rp_bucket.dtype, device=rp_bucket.device).view(-1, 1)
                    _ctx_rp_bucket_flatten = (rp_bucket + offset).flatten()

                    embs = torch.index_select(lookup_table.flatten(2), 2, _ctx_rp_bucket_flatten).\
                            view(B, -1, L, L)
        else:
            raise ValueError

        return embs

    def forward(self, temporal_tokens, query_emb=None):
    
        B,L = temporal_tokens.shape

        rp_bucket = temporal_tokens.unsqueeze(-1) - temporal_tokens.unsqueeze(1)
        if self.relative_directional:
            rp_bucket += (self.num_buckets//2)
        else:
            rp_bucket = torch.abs(rp_bucket)

        if not self.in_lin:
            rp_bucket = rp_bucket[0]
            embs = self._meta_forward(rp_bucket, query_emb)
        else:
            _embs = []
            for bs in range(int(B)):
                _rp_bucket = rp_bucket[bs]
                _q_e = query_emb[bs:bs+1] if query_emb is not None else None
                emb = self._meta_forward(_rp_bucket, _q_e)
                _embs.append(emb)

            embs = torch.cat(_embs, dim=0)

        return embs # B,1,L,L 


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.rpe_q, self.rpe_k, self.rpe_v = None, None, None

        if config.enable_relative:
            _cfg = {'relative_mode': 'contextual', 
                    'relative_directional': True, 
                    'relative_multihead': True,
                    'num_buckets': config.max_n_len,
                    'in_lin':False}

            self.rpe_q = RelativeTemEmbedding(config, **_cfg)
            self.rpe_k = RelativeTemEmbedding(config, **_cfg)

        self.config = config

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask, temporal_tokens=None):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)

        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # * Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # * add contexual relative position embedding
        if self.rpe_k is not None:
            attention_scores += self.rpe_k(temporal_tokens, query_layer)
        if self.rpe_q is not None:
            attention_scores += self.rpe_q(temporal_tokens, key_layer / math.sqrt(self.attention_head_size)).transpose(2, 3)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        # * add contexual relative position embedding
        if self.rpe_v is not None:
            context_layer += self.rpe_v(temporal_tokens, attention_probs) 

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer 


class BertSelfAttentionSen(nn.Module):
    def __init__(self, config):
        super(BertSelfAttentionSen, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.rpe_q, self.rpe_k, self.rpe_v = None, None, None

        if config.enable_relative:
            _cfg = {'relative_mode': 'contextual', 
                    'relative_directional': True, 
                    'relative_multihead': True,
                    'num_buckets': config.conf_bucket_size,
                    'in_lin': True}

            self.rpe_q = RelativeTemEmbedding(config, **_cfg)
            self.rpe_k = RelativeTemEmbedding(config, **_cfg)

        self.config = config

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask, confidence_vector=None):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)

        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # * Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # * add contexual relative position embedding
        # ? rpe_q and rpe_k would be None if not contextual
        if self.rpe_k is not None:
            attention_scores += self.rpe_k(confidence_vector, query_layer)
        if self.rpe_q is not None:
            attention_scores += self.rpe_q(confidence_vector, key_layer / math.sqrt(self.attention_head_size)).transpose(2, 3)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        # * add contexual relative position embedding
        if self.rpe_v is not None:
            context_layer += self.rpe_v(confidence_vector, attention_probs) 

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer 


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, _func_attn=BertSelfAttention):
        super(BertAttention, self).__init__()
        self.self = _func_attn(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, temporal_tokens=None, confidence_vector=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        inputs = {'query_states':input_tensor, 'key_states':input_tensor, 'value_states':input_tensor, 'attention_mask':attention_mask}
        if temporal_tokens is not None:
            inputs['temporal_tokens'] = temporal_tokens
        if confidence_vector is not None:
            inputs['confidence_vector'] = confidence_vector
        self_output = self.self(**inputs)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        """(N, L, D)"""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights=None):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        if config.share_wd_cls_weight:
            assert bert_model_embedding_weights is not None, \
                "bert_model_embedding_weights should not be None " \
                "when setting --share_wd_cls_weight flag to be true"
            assert config.hidden_size == bert_model_embedding_weights.size(1), \
                "hidden size has be the same as word embedding size when " \
                "sharing word embedding weight and classifier weight"
            self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                     bert_model_embedding_weights.size(0),
                                     bias=False)
            self.decoder.weight = bert_model_embedding_weights
        else:
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        """(N, L, D)"""
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states  # (N, L, vocab_size)

