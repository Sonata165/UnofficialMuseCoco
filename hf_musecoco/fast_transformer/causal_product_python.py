'''
Python re-implementation of causal_product_cpu.cpp and causal_product.py

Author: Longshen Ou
'''

import torch

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



