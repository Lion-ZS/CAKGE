import os
import argparse
import random

import torch
import numpy as np
from load_data import DataLoader
from base_model import IndudctiveTrainer
from utils import select_gpu
from models.CAKGE import CAKGE
from pprint import pprint
from utils import Dict
parser = argparse.ArgumentParser(description="Parser for CAKGE")
parser.add_argument('--data_path', type=str, default='data/WN18RR_v1')
parser.add_argument('--seed', type=str, default=1234)
parser.add_argument('--gpu', type=str, default=0)
args = parser.parse_args()

class Options(object):
    def dict(self):
        return self.__dict__
    pass

def set_seed(seed):
    SEED = seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)  
    torch.cuda.manual_seed(SEED)  
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    # if benchmark=True, deterministic will be False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]

    opts = Dict()

    try:
        gpu = int(args.gpu)#select_gpu()
    except UnicodeDecodeError:
        gpu = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    print('gpu:', gpu)

    loader = DataLoader(args.data_path)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel

    if dataset == 'WN18RR_v2':
        opts.lr = 0.00308
        opts.lamb = 0.0004
        opts.decay_rate = 0.994
        opts.hidden_dim = 16
        opts.attn_dim = 1
        opts.dropout = 0.200
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 20
        opts.n_extra_layer = 2

    opts.act = 'relu'
    opts.n_extra_layer = 0
    
    my_model = CAKGE
    my_model_name = "CAKGE"
    results_dir = 'results'
    epoch_num = 60
    
    config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s,%s\n' % (
        opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout,
        opts.act, str(my_model))
    config_list = [opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim,
                   opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act]
    config_list = [str(item) for item in config_list]

    print(config_str)
    results_dir = os.path.join(results_dir, my_model_name, dataset)
    best_model_path = os.path.join(results_dir, 'best.pth')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    opts.save_path = results_dir
    opts.perf_file = os.path.join(results_dir, dataset + '_perf.txt')
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)
        pprint(opts.to_dict(),stream=f)
    model = my_model(opts, loader)
    pprint(opts.to_dict())
    
    print("\nnow use ", type(model))
    trainer = IndudctiveTrainer(opts, loader, model=model)

    best_mrr = 0
    best_dict = None
    best_str = ""
    for epoch in range(epoch_num):
        print(f"start {epoch} epoch")
        mrr, out_str, out_dict = trainer.train_batch()
        with open(opts.perf_file, 'a+') as f:
            f.write(out_str)
        if float(mrr)>1-1e-3:
            # Exclude the phenomenon of NaN caused by training failure, and give up the experiment
            metrics = {"default":float(best_mrr)}
            metrics.update(out_dict)
            break
        metrics = {"default":float(mrr)}
        metrics.update(out_dict)
        if mrr > best_mrr:
            best_mrr = mrr
            best_str = out_str
            best_dict = out_dict
            print(str(epoch) + '\t' + best_str)
            torch.save(model.state_dict(), best_model_path)

    print(best_str)
    metrics = {"default":float(best_mrr)}
    metrics.update(best_dict)
    result_list = [str(type(model))] + [str(value) for key,
                                        value in best_dict.items()] + config_list
    with open(opts.perf_file, 'a+') as f:
        f.write("Best:" + best_str)
    with open("result_statistic.md", "a") as f:
        f.write("|"+"|".join(result_list) + "|\n")
    print("save results finished")
