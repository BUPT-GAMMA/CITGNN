import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import *
from model import *
from propagation import *
from params_M import *
import warnings
# from processHdata import *

warnings.filterwarnings('ignore')
args = set_params()

dataset_str = args.dataset


adj, features, labels, idx_train, idx_val, idx_test = load_data2(dataset_str)


model_args = {
    'hiddenunits': [64],
    'drop_prob': args.dropout,
    'p': args.p,
    'epochtimes':args.epochtimes,
    'cuda_id':args.cuda_id,
    'clusters':args.clusters
    }

model = PPNP(features.shape[1], max(labels)+1, **model_args)

reg_lambda = 5e-3
learning_rate = 0.01
print_interval = 20

reg_lambda = torch.tensor(reg_lambda)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


model.cuda(args.cuda_id)
features = features.cuda(args.cuda_id)
labels = labels.cuda(args.cuda_id)
idx_train = idx_train.cuda(args.cuda_id)
idx_val = idx_val.cuda(args.cuda_id)
idx_test = idx_test.cuda(args.cuda_id)

adj_normalized = normalize_adj(adj)
adj_dense = torch.FloatTensor(np.array(adj_normalized.todense()))
adj_dense = adj_dense.cuda(args.cuda_id)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    log_preds,mc_loss, o_loss = model(adj_dense,features,0,epoch, adj)
    preds = torch.argmax(log_preds,dim=1)
    
    cross_entropy_mean = F.nll_loss(log_preds[idx_train], labels[idx_train])
    acc_train = accuracy(log_preds[idx_train], labels[idx_train])
    
    l2_reg = sum((torch.sum(param ** 2) for param in model.reg_params))
    loss_train = 0.5*(cross_entropy_mean + reg_lambda / 2 * l2_reg) + 0.3*mc_loss + 0.2*o_loss
    
    loss_train.backward()
    optimizer.step()
    
    loss_val = F.nll_loss(log_preds[idx_val], labels[idx_val])
    acc_val = accuracy(log_preds[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch+1),
      'loss_train: {:.4f}'.format(loss_train.data.item()),
      'acc_train: {:.4f}'.format(acc_train.data.item()),
      'loss_val: {:.4f}'.format(loss_val.data.item()),
      'acc_val: {:.4f}'.format(acc_val.data.item()),
      'time: {:.4f}s'.format(time.time() - t))

def compute_test():
    model.eval()

    graph_test = ['_add_0.5']
    for i in graph_test:
        adj_add = sp.load_npz(f"../data/{dataset_str}-graph/{dataset_str}{i}.npz")
        output = model(adj_dense,features, 1, 0,adj_add)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        f1 = F1score(output[idx_test].cpu(),labels[idx_test].cpu())
        print("accuracy= {:.4f}".format(acc_test.item()), "f1-score={:.4f}".format(f1.item()),'\n')


t_total = time.time()
for epoch in range(500):
    train(epoch)

compute_test()
