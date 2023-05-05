# -*-  coding: utf-8 -*-
# @Time      :2021/3/1 20:58
# @Author    :huangzg28153
# @File      :train_step.py
# @Software  :PyCharm

import sys
from pyHGT.data import *
from pyHGT.model import *
from warnings import filterwarnings
filterwarnings("ignore")
import multiprocessing as mp
import time
import argparse


parser = argparse.ArgumentParser(description='Training GNN on Paper-Field (L2) classification task')

"""
1，首先读取数据，然后处理为data.Graph()格式；
2，读取文件，然后抽取子图,sample_subgraph,sample_subgraph调用feature_OAG抽取特征；
3，将抽取的子图转换为tensor张量，进行训练
4，validation；
"""
'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./data/oag_output',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./model_save',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='PF',
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
"""
图数据生成
"""

graph = renamed_load(open(os.path.join(args.data_dir, 'graph%s.pk' % args.domain), 'rb'))
train_range = {t: True for t in graph.times if t != None and t < 2015}
valid_range = {t: True for t in graph.times if t != None and t >= 2015  and t <= 2016}
test_range  = {t: True for t in graph.times if t != None and t > 2016}

types = graph.get_types()
'''
    cand_list stores all the L2 fields, which is the classification domain.
    cand_list, 保存所有PF_in_L2下的target_id，即field_id
'''
cand_list = list(graph.edge_list['field']['paper']['PF_in_L2'].keys())


def node_classification_sample(seed, pairs, time_range):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers), get their time.
        用于构造节点分类样本

        pairs : {target_id:[[source_ids,...], time]},特定关系下的节点字典，此处为rev_PF_in_L2；
        time_range: 时间范围
        首先从pair中采样一批节点ID,然后构造一个target_info列表，其元素为[target_id,_time]
        然后根据target_info构造inp,inp只包含“paper"一种类型，采样一个子图，


    '''
    np.random.seed(seed)
    # 从rev_PF_in_L2构造的pairs字典中随机抽样batch_size个，target_ids，paper在原图的id；
    target_ids = np.random.choice(list(pairs.keys()), args.batch_size, replace=False)
    target_info = []
    # 用于构造 inp, inp只包含"paper"类型，inv_PF_L2下的target节点，pair[target_id] = [[source_id],time]
    # 因此inp只包含原图target_id及时间
    # paper节点信息列表,target_info;
    for target_id in target_ids:
        _, _time = pairs[target_id]
        target_info += [[target_id, _time]]

    '''
        (2) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'
        # 抽样batch_size个节点的子图，返回特征,新体系下的边列表；
    '''
    feature, times, edge_list, _, _ = sample_subgraph(graph,
                                                      sampled_depth=args.sample_depth,
                                                      inp={'paper': np.array(target_info)},
                                                      sampled_number=args.sample_width)

    '''
        (3) Mask out the edge between the output target nodes (paper) with output source nodes (L2 field)
        掩码”paper"-"field"-"PF_in_L2"和"filed"-"paper"-"rev_PF_in_L2"边，batch掩码，
        删除papaer的ser小于batch_size所构成的边；
    '''
    masked_edge_list = []
    for i in edge_list['paper']['field']['rev_PF_in_L2']:
        # i,[target_ser,source_ser],新体系下的节点index
        if i[0] >= args.batch_size:
            # 只保留target_ser大于batch_size的节点，即paper的ser大于batch_size的节点
            masked_edge_list += [i]
    edge_list['paper']['field']['rev_PF_in_L2'] = masked_edge_list

    masked_edge_list = []
    for i in edge_list['field']['paper']['PF_in_L2']:
        # 只保留paper的ser大于batch_size的节点
        if i[1] >= args.batch_size:
            masked_edge_list += [i]
    edge_list['field']['paper']['PF_in_L2'] = masked_edge_list

    '''
        (4) Transform the subgraph into torch Tensor (edge_index is in format of pytorch_geometric)
    '''
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
        to_torch(feature, times, edge_list, graph)
    '''
        (5) Prepare the labels for each output target node (paper), and their index in sampled graph.
            (node_dict[type][0] stores the start index of a specific type of nodes)
    '''
    # ylabel,
    # cand_list，保存在PF_in_L2的所有target_id节点，即原图的field_id
    ylabel = np.zeros([args.batch_size, len(cand_list)])
    # target_ids，抽样的batch_size个原图中的paper_id;
    for x_id, target_id in enumerate(target_ids):
        # pairs,rev_PF_in_L2的sel_train_pair,
        # 故pairs[target_id][0]即原图paper_id对应的所有field_id
        for source_id in pairs[target_id][0]:
            # 从cand_list中搜寻source_id的index,然后令ylabel[x_id]中对应的index值为1
            ylabel[x_id][cand_list.index(source_id)] = 1
            # 求ylabel的列归一化值
    ylabel /= ylabel.sum(axis=1).reshape(-1, 1)
    # node_dict[type] = [node_num, len(node_dict)]???
    x_ids = np.arange(args.batch_size) + node_dict['paper'][0]
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel

def prepare_data(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
        以多线程的方式构造节点分类样本，
    '''
    jobs = []
    for batch_id in np.arange(args.n_batch):
        p = pool.apply_async(node_classification_sample,
                             args=(randint(), sel_train_pairs, train_range))
        jobs.append(p)
    p = pool.apply_async(node_classification_sample,
                         args=(randint(), sel_valid_pairs, valid_range))
    jobs.append(p)
    return jobs

#####################################################################################
# 所有的PF rev_PF_in_L2下的节点，构造为node_pair[target_id] = [source_id,time]
train_pairs = {}
valid_pairs = {}
test_pairs = {}
'''
    Prepare all the source nodes (L2 field) associated with each target node (paper) as dict
    train_pairs[target_id] = [[source_ids,...],_time],train、val、test集合的节点对字典
    rev_PF_in_L2关系下的节点对字典，target-paper;source-filed
    
'''
for target_id in graph.edge_list['paper']['field']['rev_PF_in_L2']:
    for source_id in graph.edge_list['paper']['field']['rev_PF_in_L2'][target_id]:
        _time = graph.edge_list['paper']['field']['rev_PF_in_L2'][target_id][source_id]
        if _time in train_range:
            if target_id not in train_pairs:
                train_pairs[target_id] = [[], _time]
            train_pairs[target_id][0] += [source_id]
        elif _time in valid_range:
            if target_id not in valid_pairs:
                valid_pairs[target_id] = [[], _time]
            valid_pairs[target_id][0] += [source_id]
        else:
            if target_id not in test_pairs:
                test_pairs[target_id] = [[], _time]
            test_pairs[target_id][0] += [source_id]

np.random.seed(43)
'''
    Only train and valid with a certain percentage of data, if necessary.
    pairs : {target_id:[source_id,time]}
    对train_pairs,val_pairs，按照data_percentage参数随机删去一部分数据
'''
sel_train_pairs = {p: train_pairs[p] for p in np.random.choice(
    list(train_pairs.keys()), int(len(train_pairs) * args.data_percentage), replace=False)
                   }
sel_valid_pairs = {p: valid_pairs[p] for p in np.random.choice(
    list(valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage), replace=False)
                   }

# ############################ 调用node_classification_sample ################# #
pool = mp.Pool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)
valid_data = jobs[-1].get()