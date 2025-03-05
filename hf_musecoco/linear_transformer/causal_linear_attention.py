#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement causally masked linear attention."""

import torch
from torch.nn import Module

from fast_transformers.causal_product import causal_dot_product
from fast_transformers.feature_maps import elu_feature_map


def causal_linear(Q, K, V):
    dtype = Q.dtype
    Q = Q.permute(0, 2, 1, 3).float().contiguous()  # [bs, n_head, seq_len, d_hidden]
    K = K.permute(0, 2, 1, 3).float().contiguous()
    V = V.permute(0, 2, 1, 3).float().contiguous()
    V_new = causal_dot_product(Q, K, V)
    # This line produce number larger than 100k.
    # Cause overflow in the line below
    # V_new[V_new>65504] = 65504      # Added by Longshen, but problem persist
    # ret = torch.log()  # [bs, seq_len, n_head, d_hidden]
    ret = V_new.permute(0, 2, 1, 3)
    ret = ret.type(dtype).contiguous()
    return ret  # original: return V_new.permute(0, 2, 1, 3).type(dtype).contiguous()


def causal_linear_fp32(Q, K, V):
    Q = Q.permute(0, 2, 1, 3).float().contiguous()  # [bs, n_head, seq_len, d_hidden]
    K = K.permute(0, 2, 1, 3).float().contiguous()
    V = V.permute(0, 2, 1, 3).float().contiguous()
    V_new = causal_dot_product(Q, K, V)
    # This line produce number larger than 100k.
    # Cause overflow in the line below
    # V_new[V_new>65504] = 65504      # Added by Longshen, but problem persist
    # ret = torch.log()  # [bs, seq_len, n_head, d_hidden]
    ret = V_new.permute(0, 2, 1, 3)
    ret = ret.float().contiguous()
    return ret  # original: return V_new.permute(0, 2, 1, 3).type(dtype).contiguous()


class CausalLinearAttention(Module):
    """Implement causally masked attention using dot product of feature maps in
    O(N D^2) complexity.

    See fast_transformers.attention.linear_attention.LinearAttention for the
    general concept of replacing the softmax with feature maps. In addition to
    that, we also make use of the fact that causal masking is a triangular mask
    which allows us to apply the masking and still compute the attention in O(N
    D^2) complexity.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, query_dimensions, feature_map=None, eps=1e-6):  # orig eps: 1e-6
        super(CausalLinearAttention, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps

    def forward(self, queries, keys, values, attn_mask=None, key_padding_mask=None):
        '''
        This layer produce nan in fp16 training.
        In debugging.
        :param queries:
        :param keys:
        :param values:
        :param attn_mask:
        :param key_padding_mask:
        :return:
        '''

        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)  # [bs, seq_len, n_head, d_hidden]
        K = self.feature_map.forward_keys(keys)

        assert attn_mask is None, "Cannot assign attn_mask for %s" % self.__class__.__name__

        if key_padding_mask is not None:
            K = K * key_padding_mask.type(queries.dtype)[:, :, None, None]

        # Compute the normalizers
        # Original code
        Z = 1 / (torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps) # Z.max() become inf in fp16

        # # Longshen: einsum can produce inf in fp16.   (Q.max()=79)
        # t = torch.einsum("nlhi,nlhi->nlh", Q.float(), K.float().cumsum(1))
        # Z = 1 / (t + self.eps)  # in fp16, t can sometimes be 0, Z.max() become inf in fp16
        # clip_value = 1  # 1024
        # Z = torch.clamp(Z, min=-clip_value, max=clip_value)

        # Original code: Compute the unnormalized result
        V = causal_linear(
            Q,
            K,
            values
        )           # This line produce inf
        ret = V * Z[:, :, :, None]  # This line produce nan in fp16

        # # Longshen: need to ensure the below operation is done in fp32
        # dtype = Q.dtype
        # V = causal_linear(
        #     Q.float(),
        #     K.float(),
        #     values.float(),
        # )

        # # Longshen: Then clip the attention score so that within range of fp16
        # ret = V * Z[:, :, :, None]  # This line produce nan in fp16
        # clip_value = 16
        # ret = torch.clamp(ret, min=-clip_value, max=clip_value)
        # ret = ret.type(dtype)  # convert back to fp16

        return ret
