# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
""" 
preprocess处理一张图,构造特征、邻接矩阵、注意力矩阵、相对距离矩阵、出度、入度、最短路径特征
My*Dataset类对每一张图进行处理；
"""
import torch
import numpy as np
import torch_geometric.datasets
from ogb.graphproppred import PygGraphPropPredDataset
# from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos

# 如果是一维的，feature_num=1,feature_offset=1+[0]=[1],故x=x+1;
# 否则,feature_num等于输入x的第1维的长度,feature_offset=[1,512+1,...,512*(feature_num-1)+1], x=x+feature_offset;
# 此处应该是限制x最多为二维，
def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x # x_i =  [x_{i,j} +1+512*j ,j in range(feature_num) ]

# 对一张图进行预处理；
def preprocess_item(item, virtual_token=True):
    # * 本模型处理的都是mol类数据集，其边特征与节点特征都是示性变量，代表对应维度特征的强度
    # * 因此可以进行对应的嵌入
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x) # ?在第1维的每个维数上增加1+512*j, for j in x.size(1) ,作用？

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # Floyd算法又称为插点法，是一种利用动态规划的思想寻找给定的加权图中多源点之间最短路径的算法，与Dijkstra算法类似。
    # Floyd算法与Dijkstra算法的不同

    # 1.Floyd算法是求任意两点之间的距离，是多源最短路，而Dijkstra(迪杰斯特拉)算法是求一个顶点到其他所有顶点的最短路径，是单源最短路。
    # 2.Floyd算法属于动态规划，我们在写核心代码时候就是相当于推dp状态方程，Dijkstra(迪杰斯特拉)算法属于贪心算法。
    # 3.Dijkstra(迪杰斯特拉)算法时间复杂度一般是o(n^2),Floyd算法时间复杂度是o(n^3),Dijkstra(迪杰斯特拉)算法比Floyd算法块。
    # 4.Floyd算法可以算带负权的，而Dijkstra(迪杰斯特拉)算法是不可以算带负权的。并且Floyd算法不能算负权回路。
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    # 最大最短路径路径；
    max_dist = np.amax(shortest_path_result)
    # edge feature here
    # 如果edge_attr维数为1，则将其变为[edge_num,1]的矩阵；
    if edge_attr is not None:

        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]

        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        # edge_num,edge_num,edge_attr.size(-1)
        # ? 例如，如果edge_attr为一维，则edge_attr_{i} + 1 + 1?
        attn_edge_type[edge_index[0, :], edge_index[1, :]
                    ] = convert_to_single_emb(edge_attr) + 1

        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        item.attn_edge_type = attn_edge_type
        item.edge_input = torch.from_numpy(edge_input).long()
    # 任意两点间的最短距离
    rel_pos = torch.from_numpy((shortest_path_result)).long()
    # 在注意力矩阵上加入cls token;
    if virtual_token:
        attn_bias = torch.zeros(
            [N + 1, N + 1], dtype=torch.float)  # with graph token
    else:
        attn_bias = torch.zeros(
            [N , N], dtype=torch.float)  # without graph token
    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.rel_pos = rel_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    
    return item


class MyGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        super(MyGraphPropPredDataset, self).download()

    def process(self):
        super(MyGraphPropPredDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


# class MyPygPCQM4MDataset(PygPCQM4MDataset):
#     def download(self):
#         super(MyPygPCQM4MDataset, self).download()

#     def process(self):
#         super(MyPygPCQM4MDataset, self).process()

#     def __getitem__(self, idx):
#         if isinstance(idx, int):
#             item = self.get(self.indices()[idx])
#             item.idx = idx
#             return preprocess_item(item)
#         else:
#             return self.index_select(idx)


class MyZINCDataset(torch_geometric.datasets.ZINC):
    def download(self):
        super(MyZINCDataset, self).download()

    def process(self):
        super(MyZINCDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


if __name__ == "__main__":
    dataset = MyGraphPropPredDataset(name='ogbg-molpcba',root='dataset/')
    item = dataset.__getitem__(2)