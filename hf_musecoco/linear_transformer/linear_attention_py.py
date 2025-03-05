'''
A copy of the causal_linear_attention.py
that adopt the python version of linear attention
'''

"""Implement causally masked linear attention."""

import torch
from torch.nn import Module


def causal_dot_product(queries, keys, values):
    '''
    v4
    初步通过正确性检测
    '''
    # 计算所有键与值的外积
    # keys: [N, H, L, E], values: [N, H, L, M]
    # 我们希望得到一个 [N, H, L, E, M] 的tensor
    kv = torch.einsum('nhle,nhlm->nhlem', keys, values)

    # 计算累积和，模拟因果关系，使得每个位置只能访问其之前的所有位置
    # kv: [N, H, L, E, M]
    kv_cumulative = torch.cumsum(kv, dim=2)

    # 使用queries对每个位置的累积kv矩阵进行点乘，得到最终的输出
    # queries: [N, H, L, E], kv_cumulative: [N, H, L, E, M]
    # 输出 product 应为 [N, H, L, M]
    product = torch.einsum('nhle,nhlem->nhlm', queries, kv_cumulative)

    return product

class FeatureMap(Module):
    """Define the FeatureMap interface."""
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.

        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner


class ActivationFunctionFeatureMap(FeatureMap):
    """Define a feature map that is simply an element-wise activation
    function."""
    def __init__(self, query_dims, activation_function):
        super().__init__(query_dims)
        self.activation_function = activation_function

    def new_feature_map(self, device):
        return

    def forward(self, x):
        return self.activation_function(x)


elu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.elu(x) + 1
)

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


class CausalLinearAttentionPy(Module):
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
        super(CausalLinearAttentionPy, self).__init__()
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
