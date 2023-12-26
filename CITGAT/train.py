from __future__ import division
from __future__ import print_function

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
from model import GAT
from sklearn.metrics import f1_score
from params_M import *
import warnings



# Training settings

warnings.filterwarnings('ignore')
args = set_params()

dataset_str = args.dataset


adj, features, labels, idx_train, idx_val, idx_test = load_data2(dataset_str)


hidden = 8
nb_heads = 8
epochs = 400
lr = 0.01
weight_decay = 5e-4
alpha = 0.2




model = GAT(nfeat=features.shape[1], 
            nhid=hidden, 
            nclass=int(labels.max()) + 1, 
            dropout=args.dropout, 
            nheads=nb_heads, 
            alpha=alpha,
            p=args.p,
            epochtimes=args.epochtimes,
            clusters=args.clusters,
            cuda_id=args.cuda_id )


optimizer = optim.Adam(model.parameters(), 
                       lr=lr, 
                       weight_decay=weight_decay)
cuda = torch.cuda.is_available()
if cuda:
    model.cuda(args.cuda_id)
    features = features.cuda(args.cuda_id)
    adj = adj.cuda(args.cuda_id)
    labels = labels.cuda(args.cuda_id)
    idx_train = idx_train.cuda(args.cuda_id)
    idx_val = idx_val.cuda(args.cuda_id)
    idx_test = idx_test.cuda(args.cuda_id)

features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output ,mc_loss, o_loss= model(features, adj,epoch,test=0)

    loss_train = 0.5*F.nll_loss(output[idx_train], labels[idx_train]) +  0.3*mc_loss + 0.2*o_loss
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()


    loss_val = F.nll_loss(output[idx_val], labels[idx_val]) +  mc_loss + o_loss
    acc_val = accuracy(output[idx_val], labels[idx_val])
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
        adj_add = normalize_adj(adj_add)
        adj_add = torch.FloatTensor(np.array(adj_add.todense()))
        adj_add = adj_add.cuda(args.cuda_id)
        output = model(features, adj_add,0,test=1)

        acc_test = accuracy(output[idx_test], labels[idx_test])
        f1 = F1score(output[idx_test].cpu(),labels[idx_test].cpu())
  
        print("accuracy= {:.4f}".format(acc_test.item()), "f1-score={:.4f}".format(f1.item()),'\n')

t_total = time.time()

for epoch in range(epochs):
    train(epoch)

# Testing
compute_test()
