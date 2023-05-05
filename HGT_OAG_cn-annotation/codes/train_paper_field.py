
from pyHGT.data import *
from pyHGT.model import *
from warnings import filterwarnings

filterwarnings("ignore")
import multiprocessing as mp
import time
import argparse

print("开始处理流程")
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
parser.add_argument('--cuda', type=int, default=1,
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

graph = renamed_load(open(os.path.join(args.data_dir, 'graph%s.pk' % args.domain), 'rb'))

train_range = {t: True for t in graph.times if t != None and t < 2015}
valid_range = {t: True for t in graph.times if t != None and t >= 2015  and t <= 2016}
test_range  = {t: True for t in graph.times if t != None and t > 2016}

types = graph.get_types()
'''
    cand_list stores all the L2 fields, which is the classification domain.
    cand_list,target_type "field",下所有target_id，即paper_id；
'''
cand_list = list(graph.edge_list['field']['paper']['PF_in_L2'].keys())
'''
Use KL Divergence here, since each paper can be associated with multiple fields.
Thus this task is a multi-label classification.
'''
criterion = nn.KLDivLoss(reduction='batchmean')


def node_classification_sample(seed, pairs, time_range):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers), get their time.
        用于构造节点分类样本

        pairs : {target_id:[[source_ids,...],time]}
        time_range: 时间范围
        首先从pair中采样一批节点ID,然后构造一个target_info列表，其元素为[target_id,_time]
        然后根据target_info构造inp,inp只包含“paper"一种类型，采样一个子图，
    '''
    np.random.seed(seed)
    target_ids = np.random.choice(list(pairs.keys()), args.batch_size, replace = False)
    target_info = []
    for target_id in target_ids:
        _, _time = pairs[target_id]
        target_info += [[target_id, _time]]

    '''
        (2) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'
    '''
    feature, times, edge_list, _, _ = sample_subgraph(graph,
                                                      sampled_depth=args.sample_depth,
                                                      inp={'paper': np.array(target_info)},
                                                      sampled_number=args.sample_width)


    '''
        (3) Mask out the edge between the output target nodes (paper) with output source nodes (L2 field)
        掩码”paper"-"field"-"PF_in_L2"边；edge_list
        paper_new_id 在关系rev_PF_in_L2下的边全部删除
        paper_new_id在关系PF_in_L2下的边全部删除；
        如此则删除所有target node出发的边；
    '''
    masked_edge_list = []
    for i in edge_list['paper']['field']['rev_PF_in_L2']:
        if i[0] >= args.batch_size:
            masked_edge_list += [i]
    edge_list['paper']['field']['rev_PF_in_L2'] = masked_edge_list

    masked_edge_list = []
    for i in edge_list['field']['paper']['PF_in_L2']:
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
    ylabel = np.zeros([args.batch_size, len(cand_list)])
    for x_id, target_id in enumerate(target_ids):
        for source_id in pairs[target_id][0]:
            # 从cand_list中搜寻source_id的index,然后令ylabel[x_id]中对应的index值为1,
            ylabel[x_id][cand_list.index(source_id)] = 1
            # 求ylabel的列归一化值,作为连接概率值的标签
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
        p = pool.apply_async(node_classification_sample, args=(randint(), \
            # sel, select;
            sel_train_pairs, train_range))
        jobs.append(p)
    p = pool.apply_async(node_classification_sample, args=(randint(), \
            sel_valid_pairs, valid_range))
    jobs.append(p)
    return jobs


train_pairs = {}
valid_pairs = {}
test_pairs  = {}
'''
    Prepare all the souce nodes (L2 field) associated with each target node (paper) as dict
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
                test_pairs[target_id]  = [[], _time]
            test_pairs[target_id][0]  += [source_id]


np.random.seed(43)
'''
    Only train and valid with a certain percentage of data, if necessary.
    pairs : {target_id:[source_id,time]}
'''
sel_train_pairs = {p : train_pairs[p] for p in np.random.choice(
    list(train_pairs.keys()), int(len(train_pairs) * args.data_percentage),replace = False)
                   }

sel_valid_pairs = {p : valid_pairs[p] for p in np.random.choice(
    list(valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage), replace = False)
                   }

            
'''
    Initialize GNN (model is specified by conv_name) and Classifier
'''
gnn = GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature['paper']['emb'].values[0]) + 401,
          n_hid = args.n_hid, n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout,
          num_types = len(graph.get_types()), num_relations = len(graph.get_meta_graph()) + 1).to(device)
classifier = Classifier(args.n_hid, len(cand_list)).to(device)

model = nn.Sequential(gnn, classifier)


if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters())
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters())
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters())

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)

stats = []
res = []
best_val   = 0
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
    print('Data Preparation: %.1fs' % (et - st))
    
    '''
        Train (time < 2015)
    '''
    model.train()
    # 每个batch得到一个loss值，其为标量，使用.cpu().data().detach().tolist()分离
    # 记录所有batch的loss
    train_losses = []
    torch.cuda.empty_cache()
    for _ in range(args.repeat):
        for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in train_data:
            node_rep = gnn.forward(node_feature.to(device), node_type.to(device),
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
            res  = classifier.forward(node_rep[x_ids])
            loss = criterion(res, torch.FloatTensor(ylabel).to(device))
            # 根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；
            # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了。
            #
            # 其实这里还可以补充的一点是，如果不是每一个batch就清除掉原有的梯度，而是比如说两个batch再清除掉梯度，
            # 这是一种变相提高batch_size的方法，对于计算机硬件不行，但是batch_size可能需要设高的领域比较适合，
            # 比如目标检测模型的训练。
            # ————————————————
            # 版权声明：本文为CSDN博主「xiaoxifei」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
            # 原文链接：https://blog.csdn.net/xiaoxifei/article/details/83474724
            optimizer.zero_grad() 
            torch.cuda.empty_cache()
            # 只有标量才能直接使用 backward()，即loss.backward() , pytorch 框架中的各种nn.xxLoss()，
            # 得出的都是minibatch 中各结果 平均/求和 后的值。如果使用自定义的函数，得到的不是标量，
            # 则backward()时需要传入 grad_variable 参数，
            # 这一点详见博客 https://sherlockliao.github.io/2017/07/10/backward/ 。
            # ————————————————
            # 版权声明：本文为CSDN博主「DHexia」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
            # 原文链接：https://blog.csdn.net/douhaoexia/article/details/78821428
            loss.backward()
            # nn.utils.clip_grad_norm(parameters, max_norm, norm_type=2)
            # 既然在BP过程中会产生梯度消失/爆炸（就是偏导无限接近0，导致长时记忆无法更新），那么最简单粗暴的方法，
            # 设定阈值，当梯度小于/大于阈值时，更新的梯度为阈值
            #
            # Parameters:
            # parameters (Iterable[Variable]) – 一个基于变量的迭代器，会进行归一化（原文：an iterable of Variables that will have gradients normalized）
            # max_norm (float or int) – 梯度的最大范数（原文：max norm of the gradients）
            # norm_type(float or int) – 规定范数的类型，默认为L2（原文：type of the used p-norm. Can be’inf’for infinity norm）
            # ————————————————
            # 版权声明：本文为CSDN博主「Nicola-Zhang」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
            # 原文链接：https://blog.csdn.net/yangwangnndd/article/details/99110334
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            # 那么为什么optimizer.step()需要放在每一个batch训练中，而不是epoch训练中，
            # 这是因为现在的mini-batch训练模式是假定每一个训练集就只有mini-batch这样大，
            # 因此实际上可以将每一次mini-batch看做是一次训练，一次训练更新一次参数空间，因而optimizer.step()放在这里。
            # ————————————————
            # 版权声明：本文为CSDN博主「xiaoxifei」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
            # 原文链接：https://blog.csdn.net/xiaoxifei/article/details/87797935
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]
            train_step += 1
            # scheduler.step（）按照Pytorch的定义是用来更新优化器的学习率的，一般是按照epoch为单位进行更换，
            # 即多少个epoch后更换一次学习率，因而scheduler.step()放在epoch这个大循环下。
            scheduler.step(train_step)
            del res, loss
    '''
        Valid (2015 <= time <= 2016)
    '''
    model.eval()

    # 在pytorc中 model.train()用于在训练阶段，model.eval()用在验证和测试阶段，
    # 他们的区别是对于Dropout和Batch Normlization层的影响。
    # 在train模式下，dropout网络层会按照设定的参数p设置保留激活单元的概率（保留概率=p);
    # batchnorm层会继续计算数据的mean和var等参数并更新。
    # 在val模式下，dropout层会让所有的激活单元都通过，而batchnorm层会停止计算和更新mean和var，
    # 直接使用在训练阶段已经学出的mean和var值
    with torch.no_grad():
        # requires_grad = True
        # 要求计算梯度
        # requires_grad = False
        # 不要求计算梯度
        # with torch.no_grad()或者 @ torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播

        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
        res  = classifier.forward(node_rep[x_ids])
        loss = criterion(res, torch.FloatTensor(ylabel).to(device))
        
        '''
            Calculate Valid NDCG. Update the best model based on highest NDCG score.
        '''
        valid_res = []
        for ai, bi in zip(ylabel, res.argsort(descending = True)):
            valid_res += [ai[bi.cpu().numpy()]]
        valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])
        
        if valid_ndcg > best_val:
            best_val = valid_ndcg
            torch.save(model, os.path.join(args.model_dir, args.task_name + "or" + '_' + args.conv_name))
            print('UPDATE!!!')
        
        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f") % \
              (epoch, (st-et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
                    loss.cpu().detach().tolist(), valid_ndcg))
        stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
        del res, loss
    del train_data, valid_data


'''
    Evaluate the trained model via test set (time > 2016)
'''

with torch.no_grad():
    test_res = []
    for _ in range(10):
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
                    node_classification_sample(randint(), test_pairs, test_range)
        paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                    edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
        res = classifier.forward(paper_rep)
        for ai, bi in zip(ylabel, res.argsort(descending = True)):
            test_res += [ai[bi.cpu().numpy()]]
    test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    print('Last Test NDCG: %.4f' % np.average(test_ndcg))
    test_mrr = mean_reciprocal_rank(test_res)
    print('Last Test MRR:  %.4f' % np.average(test_mrr))


best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
best_model.eval()
gnn, classifier = best_model
with torch.no_grad():
    test_res = []
    for _ in range(10):
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
                    node_classification_sample(randint(), test_pairs, test_range)
        paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                    edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
        res = classifier.forward(paper_rep)
        for ai, bi in zip(ylabel, res.argsort(descending = True)):
            test_res += [ai[bi.cpu().numpy()]]
    test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    print('Best Test NDCG: %.4f' % np.average(test_ndcg))
    test_mrr = mean_reciprocal_rank(test_res)
    print('Best Test MRR:  %.4f' % np.average(test_mrr))
