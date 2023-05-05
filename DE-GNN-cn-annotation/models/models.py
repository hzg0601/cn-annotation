# from models.layers import *
from itertools import combinations
import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, TAGConv, GATConv
from models.mlp import MLP


class GNNModel(nn.Module):
    def __init__(self, layers, in_features, hidden_features, out_features, prop_depth, dropout=0.0, model_name='DE-GNN'):
        super(GNNModel, self).__init__()
        self.layers, self.in_features, self.hidden_features, self.out_features, self.model_name = layers, in_features, hidden_features, out_features, model_name
        Layer = self.get_layer_class()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        if self.model_name == 'DE-GNN':
            self.layers.append(Layer(in_channels=in_features, out_channels=hidden_features, K=prop_depth))
        elif self.model_name == 'GIN':
            self.layers.append(
                Layer(MLP(num_layers=2, input_dim=in_features, hidden_dim=hidden_features, output_dim=hidden_features)))
        else:
            self.layers.append(Layer(in_channels=in_features, out_channels=hidden_features))
        if layers > 1:
            for i in range(layers - 1):
                if self.model_name == 'DE-GNN':
                    self.layers.append(Layer(in_channels=hidden_features, out_channels=hidden_features, K=prop_depth))
                elif self.model_name == 'GIN':
                    self.layers.append(Layer(MLP(num_layers=2, input_dim=hidden_features, hidden_dim=hidden_features,
                                                 output_dim=hidden_features)))
                elif self.model_name == 'GAT':
                    self.layers.append(Layer(in_channels=hidden_features, out_channels=hidden_features, heads=8))
                else:
                    # for GCN and GraphSAGE
                    self.layers.append(Layer(in_channels=hidden_features, out_channels=hidden_features))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_features) for i in range(layers)])
        self.merger = nn.Linear(3 * hidden_features, hidden_features)
        self.feed_forward = FeedForwardNetwork(hidden_features, out_features)

    def forward(self, batch):
        # x， num_nodes*feature_dim
        # edge_index,2*num_edges,第一维节点索引
        x = batch.x
        edge_index = batch.edge_index
        # self.layers, feature_dim,emb_dim;emb_dim,emb_dim
        # ModuleList(
        #   (0): TAGConv(6, 100, K=1)
        #   (1): TAGConv(100, 100, K=1)
        # )
        for i, layer in enumerate(self.layers):
            # edge_weight = None
            # # if not layer.normalize:
            # #     edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype, device=x.device)
            # Layer(in_channels=hidden_features,out_changes_hidden=features,K=prop_depth)(x,edge_index,edge_weight=None)
            # 每个layer包括message、propogate、update三个完整的单元；
            # 每次调用layer，都更新edge_index_i，因此可以实现每调一层layer都聚合一层特征；
            x = layer(x, edge_index, edge_weight=None)
            x = self.act(x)
            x = self.dropout(x)  # [n_nodes, mini_batch, input_dim]
            if self.model_name == 'DE-GNN':
                x = self.layer_norms[i](x)
        x = self.get_minibatch_embeddings(x, batch)
        x = self.feed_forward(x)
        return x

    def get_minibatch_embeddings(self, x, batch):
        device = x.device
        # set_indices,边列表，节点索引对的形式作为一个数组,batch_size*2
        # batch,tensor([ 0,  0,  0,  ..., 63, 63, 63]),每个节点所属子图的掩码,长度为batch包含的所有节点数
        # num_graphs,batch_size
        set_indices, batch, num_graphs = batch.set_indices, batch.batch, batch.num_graphs
        # torch.eyes(num_graphs),batch_size*batch_size
        # torch.eyes(num_graphs)[batch], len(batch)*batch_size,即按照节点所属子图的掩码，抽取一个64维的向量，
        # 再求和，num_nodes长度即batch_size,其元素每个Data包含的节点数；
        num_nodes = torch.eye(num_graphs)[batch].to(device).sum(dim=0)
        zero = torch.tensor([0], dtype=torch.long).to(device)
        # torch.cumsum(num_nodes,dim=0,dtype=torch.long),[292,len(batch)]
        # index从0开始，至len(batch)的前一位结束，即每个子图，节点的顺序索引，
        # index_bases维度为64，节点数的累加和，从0开始，至(和-1)结束；
        index_bases = torch.cat([zero, torch.cumsum(num_nodes, dim=0, dtype=torch.long)[:-1]])
        # expand(a,b,c,d...)按照(a,b,c,d..)的shape扩展张量，若为-1则按原维度扩展
        # 此处即为64*2,unsqueeze,增加一个维度；
        index_bases = index_bases.unsqueeze(1).expand(-1, set_indices.size(-1))
        assert(index_bases.size(0) == set_indices.size(0))
        #
        #
        set_indices_batch = index_bases + set_indices
        # print('set_indices shape', set_indices.shape, 'index_bases shape', index_bases.shape, 'x shape:', x.shape)
        #
        #
        x = x[set_indices_batch]  # shape [Batch_size, set_size(点为1，边为2), Feature_dim]
        #
        #
        x = self.pool(x)
        return x

    def pool(self, x):
        """
        若x的第二个维度大于1，即样本为边或子图，则需要对其pooling
        :param x:
        :return:
        """
        if x.size(1) == 1:
            return torch.squeeze(x, dim=1)
        # use mean/diff/max to pool each set's representations
        x_diff = torch.zeros_like(x[:, 0, :], device=x.device)
        for i, j in combinations(range(x.size(1)), 2):
            x_diff += torch.abs(x[:, i, :]-x[:, j, :])
        x_mean = x.mean(dim=1)  # 一个数
        x_max = x.max(dim=1)[0]  # 一个数组
        # self.merger,nn.Linear(3*hidden_features,hidden_features)
        x = self.merger(torch.cat([x_diff, x_mean, x_max], dim=-1))
        return x

    def get_layer_class(self):
        layer_dict = {'DE-GNN': TAGConv, 'GIN': GINConv, 'GCN': GCNConv, 'GraphSAGE': SAGEConv, 'GAT': GATConv}  # TAGConv essentially sums up GCN layerwise outputs, can use GCN instead 
        Layer = layer_dict.get(self.model_name)
        if Layer is None:
            raise NotImplementedError('Unknown model name: {}'.format(self.model_name))
        return Layer

    def short_summary(self):
        return 'Model: {}, #layers: {}, in_features: {}, hidden_features: {}, out_features: {}'.format(self.model_name, self.layers, self.in_features, self.hidden_features, self.out_features)


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, out_features, act=nn.ReLU(), dropout=0):
        super(FeedForwardNetwork, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)
        # torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中。
        # 另外，也可以传入一个有序模块。使用torch.nn.Sequential会自动加入激励函数
        # Module 里面也可以使用 Sequential，同时 Module 非常灵活，
        # 具体体现在 forward 中，如何复杂的操作都能直观的在 forward 里面执行
        # 使用torch.nn.Module，我们可以根据自己的需求改变传播过程，如RNN等
        self.layer1 = nn.Sequential(nn.Linear(in_features, in_features), self.act, self.dropout)
        # 是用于设置网络中的全连接层的，需要注意的是全连接层的输入与输出都是二维张量，
        # 一般形状为[batch_size, size]，不同于卷积层要求输入输出是四维张量。
        self.layer2 = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x
