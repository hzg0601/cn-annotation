# -*-  coding: utf-8 -*-
# @Time      :2021/2/5 14:28
# @Author    :huangzg28153
# @File      :SplitDataList.py
# @Software  :PyCharm

import numpy as np
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split


def split_datalist(data_list, masks):
    # generate train_set
    train_mask, val_test_mask = masks
    num_graphs = len(data_list)
    assert((train_mask.sum()+val_test_mask.sum()).astype(np.int32) == num_graphs)
    assert(train_mask.shape[0] == num_graphs)
    train_indices = np.arange(num_graphs)[train_mask.astype(bool)]
    train_set = [data_list[i] for i in train_indices]
    # generate val_set and test_set
    val_test_indices = np.arange(num_graphs)[val_test_mask.astype(bool)]
    val_test_labels = np.array([data.y for data in data_list], dtype=np.int32)[val_test_indices]
    val_indices, test_indices = train_test_split(val_test_indices, test_size=int(0.5*len(val_test_indices)), stratify=val_test_labels)
    val_set = [data_list[i] for i in val_indices]
    test_set = [data_list[i] for i in test_indices]
    return train_set, val_set, test_set


def load_datasets(train_set, val_set, test_set, bs):
    num_workers = 0
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def print_dataset(dataset, logger):
    for i in range(len(dataset)):
        data = dataset[i]
        keys = ['old_set_indices', 'old_subgraph_indices', 'set_indices', 'edge_index', 'x', 'y']
        for key in keys:
            logger.info(key)
            logger.info(data.__dict__[key])