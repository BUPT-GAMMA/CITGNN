from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils_ppnp import MixedLinear, MixedDropout
from torch_geometric.nn import dense_mincut_pool
import random
from torch.nn import Linear 
from torch.nn.parameter import Parameter
import scipy.sparse as sp
from propagation import *


class PPNP(nn.Module):
    def __init__(self, nfeatures: int, nclasses: int, hiddenunits: List[int], drop_prob: float,
                p: float, epochtimes: int, clusters: int,cuda_id: int,bias: bool = False):
        super().__init__()

        fcs = [MixedLinear(nfeatures, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i - 1], hiddenunits[i], bias=bias))
        fcs.append(nn.Linear(hiddenunits[-1], nclasses, bias=bias))
        self.fcs = nn.ModuleList(fcs)

        self.reg_params = list(self.fcs[0].parameters())

        if drop_prob is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)
        self.act_fn = nn.ReLU()


        self.p = p
        self.epochtimes = epochtimes
        self.mlp = Linear(hiddenunits[0], clusters)
        self.device = torch.device(('cuda:'+str(cuda_id)) if torch.cuda.is_available() else 'cpu')

    def _transform_features(self, attr_matrix: torch.sparse.FloatTensor):
        layer_inner = self.act_fn(self.fcs[0](self.dropout(attr_matrix)))
        
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = self.fcs[-1](self.dropout(layer_inner))
        return res

    def transform_features_1(self, attr_matrix: torch.sparse.FloatTensor):
        layer_inner = self.act_fn(self.fcs[0](self.dropout(attr_matrix)))

        return layer_inner

    def transform_features_2(self, layer_inner: torch.sparse.FloatTensor):
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = self.fcs[-1](self.dropout(layer_inner))

        return res


    def DSU(self,h_embedding,h_clu,assignment_matrics):
      
        index = random.sample(range(0,h_embedding.shape[0]),int(self.p * h_embedding.shape[0]))
        tensor_mask = torch.ones(h_embedding.shape[0],1).to(self.device)
        tensor_mask[index] = 0
        tensor_selectclu = torch.randint(low=0, high=h_clu.shape[0]-1, size=(h_embedding.shape[0],)).to(self.device)
        Select = torch.argmax(assignment_matrics,dim = 1).to(self.device)
        tensor_selectclu[tensor_selectclu == Select] = h_clu.shape[0]-1
        a1 = torch.unsqueeze(h_embedding,0)
        a1 = a1.repeat(h_clu.shape[0],1,1)
        b1 = h_clu.unsqueeze(1)
        c = a1 - b1
        d = torch.pow(c,2)
        s = assignment_matrics.t()
        s = s.unsqueeze(1)
        tensor_var_clu = torch.bmm(s,d).squeeze()
        tensor_std_clu = torch.pow(tensor_var_clu + 1e-10,0.5)        
        tensor_mean_emb = h_embedding.mean(1, keepdim =True)
        tensor_std_emb = h_embedding.var(1, keepdim = True).sqrt()      
        sigma_mean = h_clu.mean(1,keepdim=True).var(0).sqrt()
        sigma_std = (tensor_std_clu.var(0)+1e-10).sqrt() 
        tensor_beta = tensor_std_clu[tensor_selectclu] + torch.randn_like(tensor_std_emb) * sigma_std
        tensor_gama = h_clu[tensor_selectclu] + torch.randn_like(tensor_std_emb) * sigma_mean
        h_new = tensor_mask * h_embedding + (1-tensor_mask) * (((h_embedding - h_clu[Select])/(tensor_std_clu[Select]+1e-10))*(tensor_beta) + tensor_gama)


        return h_new     

    def forward(self, adj_dense, attr_matrix: torch.sparse.FloatTensor, test: int, epochs: int,adj: sp.spmatrix):
        
        x = self.transform_features_1(attr_matrix)
        self.propagation = PPRPowerIteration(adj, alpha=0.1, niter=10).to(self.device)
        if test==0:
            

            assignment_matrics = self.mlp(x)
            assignment_matrics = nn.Softmax(dim = -1)(assignment_matrics)  
            _, _, mc_loss, o_loss = dense_mincut_pool(x, adj_dense, assignment_matrics)  
            h_pool = torch.matmul(torch.transpose(assignment_matrics, 0, 1),x)
            if epochs % self.epochtimes == 0:
                x = self.DSU(x,h_pool,assignment_matrics)

            
            local_logits = self.transform_features_2(x)
            final_logits = self.propagation(local_logits)

            return F.log_softmax(final_logits, dim=-1),mc_loss, o_loss
              
        else:
            local_logits = self.transform_features_2(x)
            final_logits = self.propagation(local_logits)
            return F.log_softmax(final_logits, dim=-1)

