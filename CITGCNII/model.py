import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear 
from torch_geometric.nn import dense_mincut_pool
import random




class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, variant,p,epochtimes,clusters,cuda_id=1):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()


        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

        self.p = p
        self.epochtimes = epochtimes
        self.mlp = Linear(nhidden, clusters)

        self.device = torch.device(('cuda:'+str(cuda_id)) if torch.cuda.is_available() else 'cpu')

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

    def forward(self, x, adj,adj_dense,epoch = 0,test = 0):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))

        #DSU
        if test == 0:
            assignment_matrics = self.mlp(layer_inner)
            assignment_matrics = nn.Softmax(dim = -1)(assignment_matrics)

            _, _, mc_loss, o_loss = dense_mincut_pool(layer_inner, adj_dense, assignment_matrics)  
            h_pool = torch.matmul(torch.transpose(assignment_matrics, 0, 1),layer_inner)

            if epoch % self.epochtimes == 0:
                layer_inner = self.DSU(layer_inner,h_pool,assignment_matrics)

            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)

            layer_inner = self.fcs[-1](layer_inner)
            return F.log_softmax(layer_inner, dim=1), mc_loss, o_loss

        else:

            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)

            layer_inner = self.fcs[-1](layer_inner)
            return F.log_softmax(layer_inner, dim=1)