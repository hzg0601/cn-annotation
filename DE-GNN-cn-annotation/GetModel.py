# -*-  coding: utf-8 -*-
# @Time      :2021/2/19 13:24
# @Author    :huangzg28153
# @File      :GetModel.py
# @Software  :PyCharm
import sys
import torch
import numpy as np
from models.models import GNNModel


def get_model(layers, in_features, out_features, prop_depth, args, logger):
    model_name = args.model
    if model_name in ['DE-GNN', 'GIN', 'GCN', 'GraphSAGE', 'GAT']:
        model = GNNModel(layers=layers, in_features=in_features, hidden_features=args.hidden_features,
                         out_features=out_features, prop_depth=prop_depth, dropout=args.dropout,
                         model_name=model_name)
    else:
        return NotImplementedError
    logger.info(model.short_summary())
    return model

def get_optimizer(model, args):
    optim = args.optimizer
    if optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise NotImplementedError


def estimate_storage(dataloaders, names, logger):
    total_gb = 0
    for dataloader, name in zip(dataloaders, names):
        dataset = dataloader.dataset
        storage = 0
        total_length = len(dataset)
        sample_size = 100
        for i in np.random.choice(total_length, sample_size):
            storage += (sys.getsizeof(dataset[i].x.storage()) + sys.getsizeof(dataset[i].edge_index.storage()) +
                        sys.getsizeof(dataset[i].y.storage())) + sys.getsizeof(dataset[i].set_indices.storage())
        gb = storage*total_length/sample_size/1e9
        total_gb += gb
    logger.info('Data roughly takes {:.4f} GB in total'.format(total_gb))
    return total_gb