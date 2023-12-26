from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import GCN
from sklearn.metrics import f1_score
from params_M import *
import warnings

warnings.filterwarnings('ignore')
args = set_params()

dataset_str = args.dataset

adj , features, labels, idx_train, idx_val, idx_test = load_data2(dataset_str)

hidden = 16
epochs = 400
lr = 0.01
weight_decay = 5e-4


model = GCN(nfeat=features.shape[1],
            nhid=hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            p=args.p,
            epochtimes=args.epochtimes,
            clusters=args.clusters,
            cuda_id=args.cuda_id)
optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)
cuda = torch.cuda.is_available()

if cuda:
    model.cuda(args.cuda_id)
    features = features.cuda(args.cuda_id)
    labels = labels.cuda(args.cuda_id)
    adj = adj.cuda(args.cuda_id)
    idx_train = idx_train.cuda(args.cuda_id)
    idx_val = idx_val.cuda(args.cuda_id)
    idx_test = idx_test.cuda(args.cuda_id)

torch.autograd.set_detect_anomaly(True)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, mc_loss, o_loss= model(features, adj,epoch,test = 0)
    loss_train = 0.5*F.nll_loss(output[idx_train], labels[idx_train]) + 0.3*mc_loss + 0.2*o_loss
    acc_train = accuracy(output[idx_train], labels[idx_train])
    with torch.autograd.detect_anomaly():
        loss_train.backward() 
    optimizer.step()

    loss_val = F.nll_loss(output[idx_val], labels[idx_val]) +  mc_loss + o_loss
    acc_val = accuracy(output[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    graph_test = ['_add_0.5']
    for i in graph_test:
        adj_add = sp.load_npz(f"../data/{dataset_str}-graph/{dataset_str}{i}.npz")
        adj_add = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_add)) 
        adj_add = adj_add.cuda(args.cuda_id)
        output = model(features,adj_add,0,test = 1)

        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        f1 = F1score(output[idx_test].cpu(),labels[idx_test].cpu())

        print("accuracy= {:.4f}".format(acc_test.item()), "f1-score={:.4f}".format(f1.item()),'\n')



# Train model
t_total = time.time()
for epoch in range(epochs):
    train(epoch)

test()

