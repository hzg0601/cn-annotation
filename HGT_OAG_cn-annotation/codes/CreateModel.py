# -*-  coding: utf-8 -*-
# @Time      :2021/4/1 21:17
# @Author    :huangzg28153
# @File      :CreateModel.py
# @Software  :PyCharm
import torch.nn as nn
from Args import args,device
from pyHGT.model import GNN,Classifier
import torch
import numpy as np


def create_model(in_dim=None, types=None, meta_graph=None, cand_list_len=None, args=None):
    """

    :param in_dim:
    :param types:
    :param meta_graph:
    :param cand_list_len:
    :return:
    """
    gnn = GNN(conv_name=args.conv_name,
              in_dim=in_dim,
              n_hid=args.n_hid,
              n_heads=args.n_heads,
              n_layers=args.n_layers,
              dropout=args.dropout,
              num_types=len(types),
              num_relations=len(meta_graph) + 1).to(device)
    classifier = Classifier(args.n_hid, cand_list_len).to(device)

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
    models = scheduler, model, optimizer, gnn, classifier
    return models
