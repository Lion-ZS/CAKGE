import os
import time
import torch
import queue

import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from torch import Tensor
from typing import Optional
from functools import reduce
from collections import defaultdict as ddict


import torch.nn.functional as F
from torch.autograd import Variable

SPLIT = '*' * 30


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

def degree(index, num_nodes: Optional[int] = None,
           dtype: Optional[int] = None):
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`
    """
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N, ), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)

def edge_match(edge_index, query_index):

    base = edge_index.max(dim=1)[0] + 1
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)  
    scale = scale[-1] // scale  
    edge_index_t = edge_index.t()
    query_index_t = query_index.t()
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)  
    edge_hash, order = edge_hash.sort()  
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    num_match = end - start

    offset = num_match.cumsum(0) - num_match  
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)
    return order[range], num_match

def strict_negative_mask(data, batch):

    pos_h_index, pos_t_index, pos_r_index = batch.t()
    edge_index = torch.stack([data.edge_index[0], data.edge_type])  
    query_index = torch.stack([pos_h_index, pos_r_index]) 
    edge_id, num_t_truth = edge_match(edge_index,  query_index) 
    t_truth_index = data.edge_index[1, edge_id]  
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(
        num_t_truth) 
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool,
                        device=batch.device)  
    t_mask[sample_id, t_truth_index] = 0 
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    query_index = torch.stack([pos_t_index, pos_r_index])
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)
    return t_mask, h_mask



def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask

