# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
""" 
用于将一个batch的图组装为Batch，并对batch数据进行paddle;

"""
import torch

# 对一维向量进行pad;
# * 返回一个[1,padlen]的数组
def pad_1d_unsqueeze(x, padlen):
    # ? +1的作用
    x = x + 1  # pad id = 0,?
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

# 对二维向量进行pad,对第0维进行pad,第1维的长度等于x的第一维长度，并在0维增加一个维度；
# * 返回一个[1, padlen, x.size(1)]的数组
def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0, 令最小的数为1，
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

# 先生成一个[pad,pad]的矩阵，对于第0维大于xlen的部分填充0，小于部分仍为x;对第1维大于xlen的部分填充-inf,小于部分仍为x;
# [x1, -inf,.., -inf,
#  x2, -inf,..., -inf,
# ..., -inf,..., -inf,
#  xn, -inf,..., -inf,
# 0,0, -inf,..., -inf, *(padle-xlen)
# ]       *(padle-xlen) 
# 如果xlen > padlen ，则仍为x; 最后在0维增加一个维度；
# * 返回一个[1, pad, pad]的数组
def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

# 先生成一个[pad, pad, x.size(-1)]的矩阵，对第0,1维进行填充，填充值为0；
# * 返回一个[1,pad, pad, x.size(-1)]的向量；
def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

# * 返回一个[1,pad, pad]数组，
# ? 先增加 1； 再对0,1维进行填充； 相对距离，最小值为0？

def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

# 先增加1，再对0,1,2,3进行填充，
# * 生成一个[1,pad1, pad2, pad3, xlen4]数组；
def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


class Batch():
    def __init__(self, idx, attn_bias, attn_edge_type, rel_pos, in_degree, out_degree, x, edge_input, y):
        super(Batch, self).__init__()
        self.idx = idx
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x, self.y = x, y
        self.attn_bias, self.attn_edge_type, self.rel_pos = attn_bias, attn_edge_type, rel_pos
        self.edge_input = edge_input

    def to(self, device):
        self.idx = self.idx.to(device)
        self.in_degree, self.out_degree = self.in_degree.to(
            device), self.out_degree.to(device)
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.attn_bias, self.attn_edge_type, self.rel_pos = self.attn_bias.to(
            device), self.attn_edge_type.to(device), self.rel_pos.to(device)
        self.edge_input = self.edge_input.to(device)
        return self

    def __len__(self):
        return self.in_degree.size(0)


def collator(items, max_node=512, multi_hop_max_dist=20, rel_pos_max=20, virtual_token=True):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node] 
    item0 = items[0]
    if hasattr(item0, "attn_edge_type"):
        items = [(item.idx, item.attn_bias, item.attn_edge_type, item.rel_pos, item.in_degree,
                item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y) for item in items]
        idxs, attn_biases, attn_edge_types, rel_poses, in_degrees, out_degrees, xs, edge_inputs, ys = zip(
            *items)
    else:
        items = [(item.idx, item.attn_bias, item.rel_pos, item.in_degree,
                item.out_degree, item.x, item.y) for item in items]
        idxs, attn_biases, rel_poses, in_degrees, out_degrees, xs, ys = zip(
            *items)
   
    for idx, _ in enumerate(attn_biases):
        # 令距第0个节点相对距离大于rel_pos_max的节点注意力为-inf;
        if virtual_token:
            attn_biases[idx][1:,1:][rel_poses[idx] >= rel_pos_max] = float('-inf')
        else:
            attn_biases[idx][rel_poses[idx] >= rel_pos_max] = float('-inf')
    # ? max_node_num等于所有图包含的最大节点数    
    max_node_num = max(i.size(0) for i in xs)

    y = torch.cat(ys)
    # 对每个图的特征,在节点维度进行pad;
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    # 对每个图的edge_inputs,对第1,2,3维进行pad; [1,max_node_num, max_node_num, max_dist, :]
    # 返回一个[1,max_node_num+1, max_node_num+1]且第1维超过max_node_num+1值为0、第二维超过max_node_num+1值为-inf的注意力数组，代表每一个图中的节点自注意力矩阵； 
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    # 返回一个[1,max_node_num, max_node_num]数组,代表图中每个节点间的相对距离；
    rel_pos = torch.cat([pad_rel_pos_unsqueeze(i, max_node_num)
                        for i in rel_poses])
    # 返回一个[1,max_node_num]代表每个图中节点的入度；
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])
    # 返回一个[1,max_node_num]代表每个图中节点的出度 ；
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                           for i in out_degrees])
    if hasattr(item0, "attn_edge_type") :
        # max_dist 等于所有edge_inputs的最大multi_hop_max_dist
        max_dist = max(i.size(-2) for i in edge_inputs)
        edge_input = torch.cat([pad_3d_unsqueeze(
            i, max_node_num, max_node_num, max_dist) for i in edge_inputs])

        # 返回一个[1,max_node_num, max_node_num, attn_edge_types.size(-1)]的数组，代表每个图中边类型的注意力矩阵，矩阵值为边类型的编码向量；
        attn_edge_type = torch.cat(
            [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])

        # idxs为每个图的idx构造的list； 
        return Batch(
            idx=torch.LongTensor(idxs), # idxs 图的id构造的一维索引tuple, 
            attn_bias=attn_bias,
            attn_edge_type=attn_edge_type,
            rel_pos=rel_pos,
            in_degree=in_degree,
            out_degree=out_degree,
            x=x,
            edge_input=edge_input,
            y=y,
        )
    else:
        return Batch(
            idx=torch.LongTensor(idxs),
            attn_bias=attn_bias,
            rel_pos=rel_pos,
            in_degree=in_degree,
            out_degree=out_degree,
            x=x,
            y=y,
            attn_edge_type=None,
            edge_input=None,
        )

if __name__ == "__main__":
    pass 