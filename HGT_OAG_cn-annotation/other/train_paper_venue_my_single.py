# -*-  coding: utf-8 -*-
# @Time      :2021/3/24 16:53
# @Author    :huangzg28153
# @File      :train_paper_venue_my_single.py
# @Software  :PyCharm

from pyHGT.data import *
from pyHGT.model import *
from pyHGT.SubgraphToTorch import SubgraphToTorch
from warnings import filterwarnings
filterwarnings("ignore")

import multiprocessing as mp
import time
import argparse
oaglog.logger.info("流程开始。。。")
parser = argparse.ArgumentParser(description='Training GNN on Paper-Venue (Journal) classification task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./data/oag_output',
                    help='The address of preprocessed graph.')
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
parser.add_argument('--feature_flags', type=list, default=['rw','sp'],
                    help='which kind of distance feature to use,"random walk","shortest path" or both')
parser.add_argument('--max_sprw', type=list, default=[4,4],
                    help='parameters of distance feature')
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

graph = renamed_load(open(os.path.join(args.data_dir, 'graph%s.pk' % args.domain), 'rb'))

train_range = {t: True for t in graph.times if t != None and t < 2015}
valid_range = {t: True for t in graph.times if t != None and t >= 2015 and t <= 2016}
test_range = {t: True for t in graph.times if t != None and t > 2016}

types = graph.get_types()
'''
    cand_list stores all the Journal, which is the classification domain.
'''
cand_list = list(graph.edge_list['venue']['paper']['PV_Journal'].keys())
target_relation = [['paper', 'venue', 'rev_PV_Journal'],
                   ['venue', 'paper', 'PV_Journal']]
'''
Use CrossEntropy (log-softmax + NLL) here, since each paper can be associated with one venue.
'''
criterion = nn.NLLLoss()


# @timeit_func
def node_classification_sample(seed, pairs, time_range, batch_size):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers) and their time.
    '''
    np.random.seed(seed)
    target_ids = np.random.choice(list(pairs.keys()), batch_size, replace=False)

    target_info = []
    for target_id in target_ids:
        _, _time = pairs[target_id]
        target_info += [{"paper":[[target_id, _time]]}]

    '''
        (2) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'
    '''
    # feature, times, edge_list, _, _ = sample_subgraph(graph,
    #                                                   sampled_depth=args.sample_depth,
    #                                                   inp={'paper': np.array(target_info)},
    #                                                   sampled_number=args.sample_width)

    result = SubgraphToTorch(graph,
                             inp_list=target_info,
                             time_range=time_range,
                             sampled_depth=args.sample_depth,
                             sampled_number=args.sample_width,
                             target_relation=target_relation,
                             feature_flag=args.feature_flags,
                             max_sprw=args.max_sprw
                             ).assemble_result()
    oaglog.logger.info("本批次子图抽取完成")

    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict,node_num_list = result
    '''
        (3) Mask out the edge between the output target nodes (paper) with output source nodes (L2 field)
        掩码”paper"-"field"-"PF_in_L2"边
    '''
    # masked_edge_list = []
    # for i in edge_list['paper']['field']['rev_PF_in_L2']:
    #     if i[0] >= args.batch_size:
    #         masked_edge_list += [i]
    # edge_list['paper']['field']['rev_PF_in_L2'] = masked_edge_list
    #
    # masked_edge_list = []
    # for i in edge_list['field']['paper']['PF_in_L2']:
    #     if i[1] >= args.batch_size:
    #         masked_edge_list += [i]
    # edge_list['field']['paper']['PF_in_L2'] = masked_edge_list

    '''
        (5) Prepare the labels for each output target node (paper), and their index in sampled graph.
            (node_dict[type][0] stores the start index of a specific type of nodes)
    '''
    ylabel = np.zeros([args.batch_size, len(cand_list)])
    for x_id, target_id in enumerate(target_ids):
        # 从cand_list中搜寻source_id的index,然后令ylabel[x_id]中对应的index值为1
        ylabel[x_id] = cand_list.index(pairs[target_id][0])

    # x_ids = np.arange(args.batch_size) + node_dict['paper'][0]
    x_ids = np.array(node_num_list)
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel

# @timeit_func
def prepare_data(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    # 共 n_batch+1批数据？
    for batch_id in np.arange(args.n_batch):
        p = pool.apply_async(node_classification_sample, args=(randint(), \
                                                               sel_train_pairs, train_range, args.batch_size))
        jobs.append(p)
    p = pool.apply_async(node_classification_sample, args=(randint(), \
                                                           sel_valid_pairs, valid_range, args.batch_size))
    jobs.append(p)
    return jobs


train_pairs = {}
valid_pairs = {}
test_pairs = {}
'''
    Prepare all the souce nodes (Journal) associated with each target node (paper) as dict
'''
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
'''
    Only train and valid with a certain percentage of data, if necessary.
'''
sel_train_pairs = {p: train_pairs[p] for p in
                   np.random.choice(list(train_pairs.keys()), int(len(train_pairs) * args.data_percentage),
                                    replace=False)}
sel_valid_pairs = {p: valid_pairs[p] for p in
                   np.random.choice(list(valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage),
                                    replace=False)}

'''
    Initialize GNN (model is specified by conv_name) and Classifier
'''
gnn = GNN(conv_name=args.conv_name, in_dim=len(graph.node_feature['paper']['emb'].values[0]) + 401, \
          n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout, \
          num_types=len(graph.get_types()), num_relations=len(graph.get_meta_graph()) + 1).to(device)
classifier = Classifier(args.n_hid, len(cand_list)).to(device)

model = nn.Sequential(gnn, classifier)

if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters())
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters())
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters())

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)

stats = []
res = []
best_val = 0
train_step = 1500

pool = mp.Pool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)

for epoch in np.arange(args.n_epoch) + 1:
    '''
        Prepare Training and Validation Data
    '''
    train_data = [job.get() for job in jobs[:-1]]
    valid_data = jobs[-1].get()
    pool.close()
    pool.join()
    '''
        After the data is collected, close the pool and then reopen it.
    '''
    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool)
    et = time.time()
    oaglog.logger.info('Data Preparation: %.1fs' % (et - st))

    '''
        Train (time < 2015)
    '''
    oaglog.logger.info("开始训练")
    model.train()
    train_losses = []
    torch.cuda.empty_cache()
    for _ in range(args.repeat):
        for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in train_data:
            node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
            res = classifier.forward(node_rep[x_ids])
            loss = criterion(res, ylabel.to(device))

            optimizer.zero_grad()
            torch.cuda.empty_cache()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]
            train_step += 1
            scheduler.step(train_step)
            del res, loss
    oaglog.logger.info("本epoch训练完成。")
    '''
        Valid (2015 <= time <= 2016)
    '''
    model.eval()
    with torch.no_grad():
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                               edge_time.to(device), edge_index.to(device), edge_type.to(device))
        res = classifier.forward(node_rep[x_ids])
        loss = criterion(res, ylabel.to(device))

        '''
            Calculate Valid NDCG. Update the best model based on highest NDCG score.
        '''
        valid_res = []
        for ai, bi in zip(ylabel, res.argsort(descending=True)):
            valid_res += [(bi == ai).int().tolist()]
        valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])

        if valid_ndcg > best_val:
            best_val = valid_ndcg
            torch.save(model, os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
            print('UPDATE!!!')

        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f") % \
              (epoch, (st - et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
               loss.cpu().detach().tolist(), valid_ndcg))
        stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
        del res, loss
    del train_data, valid_data
    oaglog.logger.info("本epoch验证完毕。")

'''
    Evaluate the trained model via test set (time > 2016)
'''

with torch.no_grad():
    test_res = []
    for _ in range(10):
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
            node_classification_sample(randint(), test_pairs, test_range, args.batch_size)
        paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
        res = classifier.forward(paper_rep)
        for ai, bi in zip(ylabel, res.argsort(descending=True)):
            test_res += [(bi == ai).int().tolist()]
    test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    print('Last Test NDCG: %.4f' % np.average(test_ndcg))
    test_mrr = mean_reciprocal_rank(test_res)
    print('Last Test MRR:  %.4f' % np.average(test_mrr))
oaglog.logger.info("测试集上评估完毕。")
best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
best_model.eval()
gnn, classifier = best_model
with torch.no_grad():
    test_res = []
    for _ in range(10):
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
            node_classification_sample(randint(), test_pairs, test_range, args.batch_size)
        paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
        res = classifier.forward(paper_rep)
        for ai, bi in zip(ylabel, res.argsort(descending=True)):
            test_res += [(bi == ai).int().tolist()]
    test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    print('Best Test NDCG: %.4f' % np.average(test_ndcg))
    test_mrr = mean_reciprocal_rank(test_res)
    print('Best Test MRR:  %.4f' % np.average(test_mrr))
