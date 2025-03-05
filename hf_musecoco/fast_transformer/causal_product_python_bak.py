'''
Python re-implementation of causal_product_cpu.cpp
Older versions

Author: Longshen Ou
'''

import torch

def vvt_dot(a, b):
    # Compute a * b^T
    return torch.matmul(a.unsqueeze(-1), b.unsqueeze(-2))

def vm_dot(v, m):
    # v 应为1xA维，m 应为AxB维，因此需要调整v的维度
    # 此处将v调整为1xA维
    v = v.view(1, -1)
    return torch.matmul(v, m).squeeze(0)  # 从1xB维结果中去掉多余的维度

def causal_dot_product(queries, keys, values):
    '''
    v1
    通过了初步正确性检测
    但是比较繁琐
    '''
    N, H, L, E = queries.shape
    _, _, _, M = values.shape
    product = torch.zeros_like(values)

    for n in range(N):
        for h in range(H):
            kv = torch.zeros(E, M, device=queries.device, dtype=queries.dtype)
            for l in range(L):
                kv += vvt_dot(keys[n, h, l], values[n, h, l])
                product[n, h, l] = vm_dot(queries[n, h, l], kv)

    return product

def causal_dot_product_loop_ver(queries, keys, values):
    '''
    v3   
    初步通过了正确性检测。但是for loop可能会影响速度
    '''
    N, H, L, E = queries.shape
    M = values.shape[-1]
    product = torch.zeros_like(values)

    for n in range(N):
        for h in range(H):
            kv_cumulative = torch.zeros((L, E, M), device=queries.device, dtype=queries.dtype)
            for l in range(L):
                # Calculate the outer product of keys and values at position l
                kv = torch.matmul(keys[n, h, l].unsqueeze(-1), values[n, h, l].unsqueeze(-2))
                
                # Add the current kv to the cumulative sum up to index l
                if l > 0:
                    kv_cumulative[l] = kv_cumulative[l - 1] + kv
                else:
                    kv_cumulative[l] = kv
                
                # Now perform the dot product with queries
                product[n, h, l] = torch.matmul(queries[n, h, l].unsqueeze(0), kv_cumulative[l]).squeeze(0)

    return product

def causal_dot_product(queries, keys, values):
    '''
    v4
    初步通过正确性检测
    '''
    N, H, L, E = queries.shape
    M = values.shape[-1]

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


