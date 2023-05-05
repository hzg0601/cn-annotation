# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from data import get_dataset
from lr import PolynomialDecayLR
import torch
import math
import torch.nn as nn
import pytorch_lightning as pl

from utils.flag import flag_bounded


def init_bert_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class Graphormer(pl.LightningModule):
    def __init__(
        self,
        n_layers,
        num_heads,
        hidden_dim,
        dropout_rate,
        intput_dropout_rate,
        weight_decay,
        ffn_dim,
        dataset_name,
        warmup_updates,
        tot_updates,
        peak_lr,
        end_lr,
        edge_type,
        multi_hop_max_dist,
        attention_dropout_rate,
        flag=False,
        flag_m=3,
        flag_step_size=1e-3,
        flag_mag=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_heads = num_heads
        if dataset_name == 'ZINC':
            # torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None,
            #  max_norm=None,  norm_type=2.0,   scale_grad_by_freq=False, 
            #  sparse=False,  _weight=None)
            # 其为一个简单的存储固定大小的词典的嵌入向量的查找表，意思就是说，给一个编号，
            # 嵌入层就能返回这个编号对应的嵌入向量，嵌入向量反映了各个编号代表的符号之间的语义关系。
            # num_embeddings, 词典的大小；
            # embedding_dim, 嵌入维度
            # padding_idx, 输入不足时的填充符号，网络在遇到填充id时，就不会计算其与其它符号的相关性。
            # max_norm，最大范数，超过该值时进行归一化；
            # norm_type, 范数类型，默认为2；
            # scale_grad_by_freq, 根据单词在mini-batch中的频率对梯度进行放缩，默认为False;
            # sparse, 若为True,则与权重矩阵相关的梯度转变为稀疏张量。
            
            # 节点的嵌入层； 
            self.atom_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
            # ? 边特征的嵌入层，为什么是num_heads，因为是作为偏置加入注意力矩阵
            self.edge_encoder = nn.Embedding(64, num_heads, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == 'multi_hop':
                self.edge_dis_encoder = nn.Embedding(
                    40 * num_heads * num_heads, 1)
            # ? 相对距离的嵌入层，为什么是num_heads，因为是作为偏置加入注意力矩阵；
            self.rel_pos_encoder = nn.Embedding(40, num_heads, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(
                64, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(
                64, hidden_dim, padding_idx=0)
        else:
            # ogbg-mol*输入的节点特征为九维,边的特征为三维，因此除了zinc，本模型处理的数据都是ogbg-mol类
            # 且其值均为整形，每个值代表该图中对应特征的强度表示；
            # node_idx若为实际特征，则需要增加新的embedding,但embedding是基于示性变量，显然不可能把每个节点
            # 都作为一个token， 
            self.atom_encoder = nn.Embedding(
                512 * 9 + 1, hidden_dim, padding_idx=0)
            self.edge_encoder = nn.Embedding(
                512 * 3 + 1, num_heads, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == 'multi_hop':
                self.edge_dis_encoder = nn.Embedding(
                    128 * num_heads * num_heads, 1)
            self.rel_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)
            # abs relation embedding
            self.in_degree_encoder = nn.Embedding(
                512, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(
                512, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        # transfomrer层
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)
        # 输出层
        if dataset_name == 'PCQM4M-LSC':
            self.out_proj = nn.Linear(hidden_dim, 1)
        else:
            self.downstream_out_proj = nn.Linear(
                hidden_dim, get_dataset(dataset_name)['num_class'])
        # 定义一个graph_token虚拟节点
        self.graph_token = nn.Embedding(1, hidden_dim)
        # 定义虚拟节点的距离编码，由于作为相对距离的偏置，因此维数也是num_heads; 
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.evaluator = get_dataset(dataset_name)['evaluator']
        self.metric = get_dataset(dataset_name)['metric']
        self.loss_fn = get_dataset(dataset_name)['loss_fn']
        self.dataset_name = dataset_name

        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.multi_hop_max_dist = multi_hop_max_dist
        # 手动训练的标识；
        self.flag = flag
        # 每个批次数据手动训练的次数
        self.flag_m = flag_m
        # 每个批次手动训练的学习率
        self.flag_step_size = flag_step_size
        # 每个批次手动训练的初始梯度值；如果为0，则等于学习率
        # 也是权重调整的上限，超过该值的节点需要重新赋权调整；
        self.flag_mag = flag_mag
        self.hidden_dim = hidden_dim
        self.automatic_optimization = not self.flag
        self.apply(lambda module: init_bert_params(module, n_layers=n_layers))

    def forward(self, batched_data, perturb=None):
        # attn_bias, [n_graph, n_node+1, n_node+1],原始的注意力偏置
        attn_bias, rel_pos, x = batched_data.attn_bias, batched_data.rel_pos, batched_data.x
        in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        edge_input, attn_edge_type = batched_data.edge_input, batched_data.attn_edge_type
        # graph_attn_bias
        # 
        n_graph, n_node = x.size()[:2] # x.size = [n_graph, n_node, n_dim] = [batch_size, max_num_nodes, n_dim]
        graph_attn_bias = attn_bias.clone() # [n_graph, n_node+1, n_node+1]
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        # rel pos ，[n_graph,n_node,n_node]
        # rel_pos_encoder 将每个元素映射为为num_heads长的向量，rel_pos的元素为任意两点间的最短距离，意味着n_heads > max_shortest_path?
        # 似乎不一定，one-hot才需要max_shortest_path的长度，但非one-hot则不需要；
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2)
        # 对实际节点加入位置偏置
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + rel_pos_bias
        # reset rel pos here
        # 将虚拟节点的距离编码加入节点矩阵的第0列
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        # 将虚拟节点的距离编码加入第0行
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == 'multi_hop':
            rel_pos_ = rel_pos.clone()
            rel_pos_[rel_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            rel_pos_ = torch.where(rel_pos_ > 1, rel_pos_ - 1, rel_pos_)
            if self.multi_hop_max_dist > 0:
                rel_pos_ = rel_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(
                3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                -1, self.num_heads, self.num_heads)[:max_dist, :, :])
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2) /
                          (rel_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(
                attn_edge_type).mean(-2).permute(0, 3, 1, 2)
        # 对实际节点加入边特征偏置；
        # todo 位置信息、边特征编码作为增广特征与作为边特征的区别？ 
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,:, 1:, 1:] + edge_input
        # 加入原始的注意力偏置,attn_bias.unsqueeze(1), [n_graph, 1, n_node+1, n_node+1]
        # graph_attn_bias, [n_graph, n_head, n_node+1, n_node+1]
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # node feauture + graph token
       
        node_feature = self.atom_encoder(x).sum(dim=-2) # [n_graph, n_node, n_hidden]
        # * perturb为扰动，为node_feature加入扰动?
        # 如果self.flag为True,且perturb存在，则将perturb加入特征，并据此输出预测；
        # 
        if self.flag and perturb is not None:
            node_feature += perturb
        # ? 加入出度、入度编码特征；为什么是相加不是拼接？
        # self.in_degree_encoder(in_degree), [n_graph, n_node, n_hidden]
        # todo 使用原始特征的区别？ 
        node_feature = node_feature + \
            self.in_degree_encoder(in_degree) + \
            self.out_degree_encoder(out_degree)
        # 拼接实际节点特征与graph_token特征；
        graph_token_feature = self.graph_token.weight.unsqueeze(
            0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat(
            [graph_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(graph_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias)
        output = self.final_ln(output)

        # output part
        if self.dataset_name == 'PCQM4M-LSC':
            # get whole graph rep
            output = self.out_proj(output[:, 0, :])
        else:
            output = self.downstream_out_proj(output[:, 0, :])
        return output

    def training_step(self, batched_data, batch_idx):
        if self.dataset_name == 'ogbg-molpcba':
            # 如果self.flag为False,则表示self已经训练好，直接进行预测，然后损失
            if not self.flag:
                y_hat = self(batched_data).view(-1)
                y_gt = batched_data.y.view(-1).float()
                mask = ~torch.isnan(y_gt)
                loss = self.loss_fn(y_hat[mask], y_gt[mask])
            # 否则，调用flag_bounded进行训练，需要重新加入perturb，
            else:
                y_gt = batched_data.y.view(-1).float()
                mask = ~torch.isnan(y_gt)

                def forward(perturb): return self(batched_data, perturb)
                model_forward = (self, forward)
                n_graph, n_node = batched_data.x.size()[:2]
                perturb_shape = (n_graph, n_node, self.hidden_dim)

                optimizer = self.optimizers()
                optimizer.zero_grad()
                # 如果flag为True，表示手动进行求导与反向传播更新参数；
                loss, _ = flag_bounded(model_forward, perturb_shape, y_gt[mask], optimizer, batched_data.x.device, self.loss_fn,
                                       m=self.flag_m, step_size=self.flag_step_size, mag=self.flag_mag, mask=mask)
                self.lr_schedulers().step()

        elif self.dataset_name == 'ogbg-molhiv':
            if not self.flag:
                y_hat = self(batched_data).view(-1)
                y_gt = batched_data.y.view(-1).float()
                loss = self.loss_fn(y_hat, y_gt)
            else:
                y_gt = batched_data.y.view(-1).float()
                def forward(perturb): return self(batched_data, perturb)
                model_forward = (self, forward)
                n_graph, n_node = batched_data.x.size()[:2]
                perturb_shape = (n_graph, n_node, self.hidden_dim)

                optimizer = self.optimizers()
                optimizer.zero_grad()
                # ? m反向传播训练的次数？ step_size, 学习率； mag, 初始梯度值； 
                loss, _ = flag_bounded(model_forward, perturb_shape, y_gt, optimizer, batched_data.x.device, self.loss_fn,
                                       m=self.flag_m, step_size=self.flag_step_size, mag=self.flag_mag)
                self.lr_schedulers().step()
        else:
            y_hat = self(batched_data).view(-1)
            y_gt = batched_data.y.view(-1)
            loss = self.loss_fn(y_hat, y_gt)
        # log日志
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batched_data, batch_idx):
        if self.dataset_name in ['PCQM4M-LSC', 'ZINC']:
            y_pred = self(batched_data).view(-1)
            y_true = batched_data.y.view(-1)
        else:
            y_pred = self(batched_data)
            y_true = batched_data.y
        return {
            'y_pred': y_pred,
            'y_true': y_true,
        }

    def validation_epoch_end(self, outputs):
        y_pred = torch.cat([i['y_pred'] for i in outputs])
        y_true = torch.cat([i['y_true'] for i in outputs])
        if self.dataset_name == 'ogbg-molpcba':
            mask = ~torch.isnan(y_true)
            loss = self.loss_fn(y_pred[mask], y_true[mask])
            self.log('valid_ap', loss, sync_dist=True)
        else:
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            try:
                self.log('valid_' + self.metric, self.evaluator.eval(input_dict)
                         [self.metric], sync_dist=True)
            except:
                pass

    def test_step(self, batched_data, batch_idx):
        if self.dataset_name in ['PCQM4M-LSC', 'ZINC']:
            y_pred = self(batched_data).view(-1)
            y_true = batched_data.y.view(-1)
        else:
            y_pred = self(batched_data)
            y_true = batched_data.y
        return {
            'y_pred': y_pred,
            'y_true': y_true,
            'idx': batched_data.idx,
        }

    def test_epoch_end(self, outputs):
        y_pred = torch.cat([i['y_pred'] for i in outputs])
        y_true = torch.cat([i['y_true'] for i in outputs])
        if self.dataset_name == 'PCQM4M-LSC':
            result = y_pred.cpu().float().numpy()
            idx = torch.cat([i['idx'] for i in outputs])
            torch.save(result, 'y_pred.pt')
            torch.save(idx, 'idx.pt')
            exit(0)
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        self.log('test_' + self.metric, self.evaluator.eval(input_dict)
                 [self.metric], sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            'scheduler': PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Graphormer")
        parser.add_argument('--n_layers', type=int, default=12)
        parser.add_argument('--num_heads', type=int, default=32)
        parser.add_argument('--hidden_dim', type=int, default=512)
        parser.add_argument('--ffn_dim', type=int, default=512)
        parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
        parser.add_argument('--dropout_rate', type=float, default=0.1)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--attention_dropout_rate',
                            type=float, default=0.1)
        parser.add_argument('--checkpoint_path', type=str, default='')
        parser.add_argument('--warmup_updates', type=int, default=60000)
        parser.add_argument('--tot_updates', type=int, default=1000000)
        parser.add_argument('--peak_lr', type=float, default=2e-4)
        parser.add_argument('--end_lr', type=float, default=1e-9)
        parser.add_argument('--edge_type', type=str, default='multi_hop')
        parser.add_argument('--validate', action='store_true', default=False)
        parser.add_argument('--test', action='store_true', default=False)
        parser.add_argument('--flag', action='store_true')
        parser.add_argument('--flag_m', type=int, default=3)
        parser.add_argument('--flag_step_size', type=float, default=1e-3)
        parser.add_argument('--flag_mag', type=float, default=1e-3)
        return parent_parser


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias) # q,k,v = y,y,y 
        y = self.self_attention_dropout(y)
        x = x + y # residual connection 

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x # residual connection by ffn， 两次residual connection? 
