import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
import numpy as np
import random
from torch.nn import Linear 
from torch.nn.parameter import Parameter

from torch_geometric.nn import dense_mincut_pool



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,p,epochtimes,clusters,cuda_id):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        


        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)



        self.p = p
        self.epochtimes = epochtimes
        self.mlp = Linear(nhid * nheads, clusters)
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


    def forward(self, x, adj,epoch,test=0):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        if test == 0:

            assignment_matrics = self.mlp(x)    
            assignment_matrics = nn.Softmax(dim = -1)(assignment_matrics)   

            _, _, mc_loss, o_loss = dense_mincut_pool(x, adj, assignment_matrics)    
            h_pool = torch.matmul(torch.transpose(assignment_matrics, 0, 1),x)
            
            if epoch % self.epochtimes == 0:
                x = self.DSU(x,h_pool,assignment_matrics)        

            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.out_att(x, adj))

            return F.log_softmax(x, dim=1),mc_loss, o_loss

        else:
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.out_att(x, adj))
            return F.log_softmax(x, dim=1)






class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,p,epochtimes,clusters):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

        self.p = p
        self.epochtimes = epochtimes
        self.mlp = Linear(nhid * nheads, clusters)

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




    def forward(self, x, adj,epoch,test=0):
        x = F.dropout(x, self.dropout, training=self.training)
        assert not torch.isnan(x).any()
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        assert not torch.isnan(x).any()

        if test == 0:

            assignment_matrics = self.mlp(x)    
            assignment_matrics = nn.Softmax(dim = 0)(assignment_matrics)   

            _, _, mc_loss, o_loss = dense_mincut_pool(x, adj, assignment_matrics)    
            h_pool = torch.matmul(torch.transpose(assignment_matrics, 0, 1),x)
            
            if epoch % self.epochtimes == 0:
                x = self.DSU(x,h_pool,assignment_matrics)        
            assert not torch.isnan(x).any()

            x = F.dropout(x, self.dropout, training=self.training)
            x = x + 1e-10
            assert not torch.isnan(x).any()
            x = F.elu(self.out_att(x, adj))
            assert not torch.isnan(x).any()

            return F.log_softmax(x, dim=1),mc_loss, o_loss

        else:
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.out_att(x, adj))
            return F.log_softmax(x, dim=1)


        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)

