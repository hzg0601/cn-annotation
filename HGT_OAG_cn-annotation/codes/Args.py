# -*-  coding: utf-8 -*-
# @Time      :2021/4/1 21:03
# @Author    :huangzg28153
# @File      :train_main.py
# @Software  :PyCharm

from warnings import filterwarnings
filterwarnings("ignore")
import torch
import argparse


parser = argparse.ArgumentParser(description='Training GNN on Paper-Venue (Journal) classification task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='/data/oag_output/',
                    help='The address of preprocessed graph.')
parser.add_argument('--subgraphs_dir',type=str,default='/data/sampled_subgraphs/',
                    help='The address of sampled subgraph.')
parser.add_argument('--model_dir', type=str, default='/model_save/',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='PV',
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

parser.add_argument('--feature_flags', type=tuple, default=(None, None),
                    help='which kind of distance feature to use,"random walk","shortest path" or both')
parser.add_argument('--max_sprw', type=tuple, default=(4, 4),
                    help='parameters of distance feature')
# parser.add_argument('--if_sample_mp',type=bool, default=True,
#                     help="sample subgraph with multiprocessing or not")
parser.add_argument("--use_identity_feature",type=bool,default=True,
                    help="use identity feature or not")
parser.add_argument('--sample_n_pool',type=int,default=32,
                    help="how many pools to sample subgraph")
parser.add_argument('--emb_len', type=int, default=400,
                    help="the embedding length of random embedding of node if 'node_emb' not"
                         "in graph.node_feature[_type]")
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
parser.add_argument('--n_pool', type=int, default=32,
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
ABSULUTE_DIR = '/data1/huangzg/research/pyHGT_OAG'

args.subgraphs_dir = ABSULUTE_DIR + args.subgraphs_dir
args.model_dir = ABSULUTE_DIR + args.model_dir

