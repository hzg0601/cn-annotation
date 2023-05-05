# -*-  coding: utf-8 -*-
# @Time      :2021/3/26 13:15
# @Author    :huangzg28153
# @File      :train_paper_venue_single_full.py
# @Software  :PyCharm

import oaglog
from pyHGT.data import renamed_load
from pyHGT.model import *
from pyHGT.SubgraphToTorch import SubgraphToTorch
from warnings import filterwarnings
filterwarnings("ignore")

import os
import numpy as np
import dill
from collections import defaultdict
import multiprocessing as mp
import argparse
oaglog.logger.info("流程开始。。。")
parser = argparse.ArgumentParser(description='Training GNN on Paper-Venue (Journal) classification task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./data/oag_output',
                    help='The address of preprocessed graph.')
parser.add_argument('--subgraphs_dir',type=str,default='./data/sampled_subgraphs',
                    help='The adress of sampled subgraph.')
parser.add_argument('--model_dir', type=str, default='./model_save',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='PV',
                    help='The name of the stored models and optimization results.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='_CS',
                    help='CS, Medicion or All: _CS or _Med or (empty)')
'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=4,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many nodes to be sampled per layer per type')
parser.add_argument('--feature_flags', type=tuple, default=('rw','sp'),
                    help='which kind of distance feature to use,"random walk","shortest path" or both')
parser.add_argument('--max_sprw', type=tuple, default=(4, 4),
                    help='parameters of distance feature')
parser.add_argument('--if_sample_mp',type=bool, default=True,
                    help="whether sample subgraph with multiprocessing or not")
parser.add_argument('--sample_n_pool',type=int,default=64,
                    help="how many pools to sample subgraph")
'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--data_percentage', type=float, default=1.0,
                    help='Percentage of training and validation data to use')
parser.add_argument('--n_epoch', type=int, default=200,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=4,
                    help='Number of process to sample subgraph')
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch')
parser.add_argument('--repeat', type=int, default=2,
                    help='How many time to train over a singe batch (reuse data)')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')
parser.add_argument('--clip', type=float, default=0.25,
                    help='Gradient Norm Clipping')

args = parser.parse_args()

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
for path in [args.data_dir, args.model_dir,args.subgraphs_dir]:
    if not os.path.exists(path):
        os.mkdir(path)

###############################################data_preparing#########################################################
oaglog.logger.info("读取数据开始。。。")
graph = renamed_load(open(os.path.join(args.data_dir, 'graph%s.pk' % args.domain), 'rb'))
oaglog.logger.info("读取数据完毕。")
train_range = {t: True for t in graph.times if t != None and t < 2015}
valid_range = {t: True for t in graph.times if t != None and t >= 2015 and t <= 2016}
test_range = {t: True for t in graph.times if t != None and t > 2016}

cand_list = list(graph.edge_list['venue']['paper']['PV_Journal'].keys())
target_relation = [['paper', 'venue', 'rev_PV_Journal'],
                   ['venue', 'paper', 'PV_Journal']]
types = list(graph.get_types())
# 元关系字典，最后一个为"self"类型
edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
edge_dict['self'] = len(edge_dict)

train_pairs = {}
valid_pairs = {}
test_pairs = {}

for target_id in graph.edge_list['paper']['venue']['rev_PV_Journal']:
    for source_id in graph.edge_list['paper']['venue']['rev_PV_Journal'][target_id]:
        _time = graph.edge_list['paper']['venue']['rev_PV_Journal'][target_id][source_id]
        if _time in train_range:
            if target_id not in train_pairs:
                train_pairs[target_id] = [source_id, _time]
        elif _time in valid_range:
            if target_id not in valid_pairs:
                valid_pairs[target_id] = [source_id, _time]
        else:
            if target_id not in test_pairs:
                test_pairs[target_id] = [source_id, _time]

np.random.seed(43)

sel_train_pairs = {p: train_pairs[p] for p in
                   np.random.choice(list(train_pairs.keys()), int(len(train_pairs) * args.data_percentage),
                                    replace=False)}
sel_valid_pairs = {p: valid_pairs[p] for p in
                   np.random.choice(list(valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage),
                                    replace=False)}

# 首先生成所有train、val、test样本

pairs_list = [sel_train_pairs, sel_valid_pairs, test_pairs]
time_range_list = [train_range, valid_range, test_range]
data_class_list = ["train",'val','test']
target_info_list = []

for pairs,data_class in zip(pairs_list,data_class_list):
    target_ids = list(pairs.keys())[:256]
    target_info = []
    for target_id in target_ids:
        _, _time = pairs[target_id]
        target_info += [{"paper": [[target_id, _time]]}]
    target_info_list.append(target_info)
    oaglog.logger.info("数据类型 %s 共 %d 个节点" % (data_class, len(target_info)))

sample_all_list = []


for target_info, time_range, data_class in zip(target_info_list, time_range_list,data_class_list):
    oaglog.logger.info("数据类型 %s 子图采集开始。。。" % data_class)

    sample_all  = SubgraphToTorch(graph,
                                  # types=types,
                                  edge_dict=edge_dict,
                                  time_range=time_range,
                                  # inp_list=target_info,
                                  sampled_depth=args.sample_depth,
                                  sampled_number=args.sample_width,
                                  if_sample_mp=args.if_sample_mp,
                                  sample_n_pool=args.sample_n_pool,
                                  target_relation=target_relation,
                                  feature_flag=args.feature_flags,
                                  max_sprw=args.max_sprw
                                 ).mp_sample_subgraphs(target_info_list, graph)
    oaglog.logger.info("数据类型 %s 子图采集开始完毕!" % data_class)

    sample_all_list.append(sample_all)
oaglog.logger.info("所有训练、验证、测试数据子图采集完毕。")


dill.dump(sample_all_list,
          open(args.subgraphs_dir + '/graph%s.pk' %
               args.domain + args.task_name + "sample_subgraphs", 'wb'))


def ramdom_sample_batch(result, types):
    """
    将输入的batch的结果组装起来，返回
    :return:
    """

    node_feature_list = []
    node_num_list = [0]
    node_type_list = []

    node_dict_list = []
    node_num_dict = defaultdict(lambda: [])

    edge_time_list = []
    edge_index_list = []
    edge_type_list = []

    result = np.random.choice(result, args.batch_size, replace=False)

    for num, result_sample in enumerate(result):

        # 每个节点的完整索引，为一个dataframe,本次未用到
        # indxs["sample"] = num
        # indxs_list.append(indxs)
        # node_feature 为一个array;
        node_feature_list += list(result_sample[0])
        # node_num为一个数字
        node_num_list.append(result_sample[1])
        # node_type为一个一层列表
        node_type_list += result_sample[2]
        # node_dict为类型：节点个数字典
        node_dict_list.append(result_sample[3])
        # edge_time为一个一层列表
        edge_time_list += result_sample[4]
        # edge_index为一个二层列表
        edge_index = result_sample[5]
        # 重新安排索引，每个样本以前一个样本的节点数为新索引的起始
        edge_index = [[pair[0] + np.sum(node_num_list),
                       pair[1] + np.sum(node_num_list)]
                      for pair in edge_index]

        edge_index_list += edge_index
        # edge_type为一个一层列表
        edge_type_list += result_sample[6]
        # oaglog.logger.info("%s 个节点特征生成完毕" % str(num))

    # # 用于节点重新排序，
    # indxs_list = pd.concat(indxs_list, axis=0)
    # tyep_dict = {_type:idx for idx,_type in enumerate(types)}
    # indxs_list["type"] = indxs_list.map(tyep_dict)
    # indxs_list = indxs_list.sort_values(by=['layer',"type","sample","ser"], axis=0).reset_index()

    # 计算每个类型下的节点数
    num_len = 0
    for t in types:
        # 首先确定每个类型的节点数目
        # 然后再按类型累加
        temp = [node_dict[t][2] for node_dict in node_dict_list]
        temp = np.sum(temp)
        node_num_dict[t] += [num_len, len(node_num_dict)]
        num_len += temp

    node_dict = node_num_dict
    # node_num_list的最后一个并非传入target_id的新索引，须删除
    node_num_list.pop(-1)

    node_feature = torch.FloatTensor(node_feature_list)
    node_type = torch.LongTensor(node_type_list)
    edge_time = torch.LongTensor(edge_time_list)
    edge_index = torch.LongTensor(edge_index_list).t()
    edge_type = torch.LongTensor(edge_type_list)

    ylabel = np.zeros([args.batch_size, len(cand_list)])
    for x_id, target_id in enumerate(target_ids):
        # 从cand_list中搜寻source_id的index,然后令ylabel[x_id]中对应的index值为1
        ylabel[x_id] = cand_list.index(pairs[target_id][0])

    # x_ids = np.arange(args.batch_size) + node_dict['paper'][0]
    x_ids = np.array(node_num_list)

    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel

######################### model build#############################################################################
# criterion = nn.NLLLoss()
# gnn = GNN(conv_name=args.conv_name,
#           in_dim=len(graph.node_feature['paper']['emb'].values[0]) + 401,
#           n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout, \
#           num_types=len(graph.get_types()), num_relations=len(graph.get_meta_graph()) + 1).to(device)
# classifier = Classifier(args.n_hid, len(cand_list)).to(device)
#
# model = nn.Sequential(gnn, classifier)
#
# if args.optimizer == 'adamw':
#     optimizer = torch.optim.AdamW(model.parameters())
# elif args.optimizer == 'adam':
#     optimizer = torch.optim.Adam(model.parameters())
# elif args.optimizer == 'sgd':
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# elif args.optimizer == 'adagrad':
#     optimizer = torch.optim.Adagrad(model.parameters())
#
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)

#######################train_val############################################
# stats = []
# res = []
# best_val = 0
# train_step = 1500
#
# for epoch in np.arange(args.n_epoch) + 1:
#     train_data = []
#     for i in np.arange(args.n_batch):
#         data = ramdom_sample_batch(sample_all_list[0], types)
#         train_data.append(data)
#     valid_data = ramdom_sample_batch(sample_all_list[1], types)
#     oaglog.logger.info("开始训练")
#     et = time.time()
#
#     model.train()
#     train_losses = []
#     torch.cuda.empty_cache()
#     for _ in range(args.repeat):
#         for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in train_data:
#             node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
#                                    edge_time.to(device), edge_index.to(device), edge_type.to(device))
#             res = classifier.forward(node_rep[x_ids])
#             loss = criterion(res, ylabel.to(device))
#
#             optimizer.zero_grad()
#             torch.cuda.empty_cache()
#             loss.backward()
#
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
#             optimizer.step()
#
#             train_losses += [loss.cpu().detach().tolist()]
#             train_step += 1
#             scheduler.step(train_step)
#             del res, loss
#     oaglog.logger.info("本epoch训练完成。")
#     '''
#         Valid (2015 <= time <= 2016)
#     '''
#     model.eval()
#     with torch.no_grad():
#         node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data
#         node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
#                                edge_time.to(device), edge_index.to(device), edge_type.to(device))
#         res = classifier.forward(node_rep[x_ids])
#         loss = criterion(res, ylabel.to(device))
#
#         '''
#             Calculate Valid NDCG. Update the best model based on highest NDCG score.
#         '''
#         valid_res = []
#         for ai, bi in zip(ylabel, res.argsort(descending=True)):
#             valid_res += [(bi == ai).int().tolist()]
#         valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])
#
#         if valid_ndcg > best_val:
#             best_val = valid_ndcg
#             torch.save(model, os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
#             print('UPDATE!!!')
#
#         st = time.time()
#         print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f") % \
#               (epoch, (st - et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
#                loss.cpu().detach().tolist(), valid_ndcg))
#         stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
#         del res, loss
#     del train_data, valid_data
#     oaglog.logger.info("本epoch验证完毕。")
#
# ###############################test##############################
#
#
# with torch.no_grad():
#     test_res = []
#     for _ in range(10):
#         node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
#             ramdom_sample_batch(sample_all_list[2], types)
#         paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
#                                 edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
#         res = classifier.forward(paper_rep)
#         for ai, bi in zip(ylabel, res.argsort(descending=True)):
#             test_res += [(bi == ai).int().tolist()]
#     test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
#     print('Last Test NDCG: %.4f' % np.average(test_ndcg))
#     test_mrr = mean_reciprocal_rank(test_res)
#     print('Last Test MRR:  %.4f' % np.average(test_mrr))
# oaglog.logger.info("测试集上评估完毕。")
# best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
# best_model.eval()
# gnn, classifier = best_model
#
# with torch.no_grad():
#     test_res = []
#     for _ in range(10):
#         node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
#             ramdom_sample_batch(sample_all_list[2], types)
#         paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
#                                 edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
#         res = classifier.forward(paper_rep)
#         for ai, bi in zip(ylabel, res.argsort(descending=True)):
#             test_res += [(bi == ai).int().tolist()]
#     test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
#     print('Best Test NDCG: %.4f' % np.average(test_ndcg))
#     test_mrr = mean_reciprocal_rank(test_res)
#     print('Best Test MRR:  %.4f' % np.average(test_mrr))