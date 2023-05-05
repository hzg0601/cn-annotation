'''
Description: 
version: 
Author: huangzg
LastEditors: huangzg
Date: 2021-09-13 15:50:20
LastEditTime: 2021-09-15 17:29:03
'''
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import math

# 手动进行反向传播
def flag_bounded(model_forward, perturb_shape, y, optimizer, device, criterion, m=3, step_size=1e-3, mag=1e-3, mask=None):
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    if mag > 0:
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-1, 1).to(device)
        perturb = perturb * mag / math.sqrt(perturb_shape[-1])
    else:
        perturb = torch.FloatTensor(
            *perturb_shape).uniform_(-step_size, step_size).to(device)
    # 所有的tensor都有.requires_grad属性,可以设置这个属性为True或False, 
    # 如果想改变这个属性，就调用tensor.requires_grad_()方法
    perturb.requires_grad_()
    # 设置初始loss
    out = forward(perturb).view(-1)
    if mask is not None:
        out = out[mask]
    loss = criterion(out, y)
    # ? 为什么除以m;
    # 多次重复manual_backward(loss)反向传播损失时，
    # 其梯度会累积，因此多次同批次数据更新需除以更新次数；
    loss /= m
    # 手动求导，更新参数
    for _ in range(m-1):
        # loss.backward()
        model.manual_backward(loss)
        perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        if mag > 0:
            perturb_data_norm = torch.norm(perturb_data, dim=-1).detach()
            # 将perturb_data_norm>mag构造的boolTensor转换为perturb_data数据类型的向量
            exceed_mask = (perturb_data_norm > mag).to(perturb_data)
            # 调整标准化后节点权重大于mag节点的权重，使其权重不超过mag; 
            reweights = (mag / perturb_data_norm * exceed_mask +
                         (1-exceed_mask)).unsqueeze(-1)
            perturb_data = (perturb_data * reweights).detach()

        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        out = forward(perturb).view(-1)
        if mask is not None:
            out = out[mask]
        loss = criterion(out, y)
        loss /= m

    # loss.backward()
    model.manual_backward(loss)
    optimizer.step()

    return loss, out
