"""
models.py
Multitask and singletask model constructors for DeepND 
Bilkent University, Department of Computer Engineering
Ankara, 2020
"""
import pandas as pd
import numpy as np
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.autograd import Variable

class DeepND_ST(torch.nn.Module): # Single Task DeepND
    def __init__(self, featsize=20, unit=15):
        super(DeepND_ST, self).__init__()
        self.unit = unit
        self.gcn_k = 1
        
        self.pfcnet = nn.ModuleList()
        self.mdcbcnet = nn.ModuleList()
        self.v1cnet = nn.ModuleList()
        self.shanet = nn.ModuleList()
            
        for i in range(network_count):
            # PFC
            layers = nn.ModuleList()
            for j in range(self.gcn_k):
                layers.append(GCNConv(self.unit, h1, improved= True, cached=True) )
                layers.append(nn.BatchNorm1d(h1,track_running_stats=False) )
                if self.gcn_k > 1:
                    layers.append(GCNConv(h1, h2, improved= True, cached=True) )
                else:
                    layers.append(GCNConv(h1, 2, improved= True, cached=True) )
            if self.gcn_k > 1:
                layers.append(nn.Linear(self.gcn_k * h2, int(self.gcn_k * h2 / 2)) )
                layers.append(nn.Linear(int(self.gcn_k * h2 / 2), 2) )
                
            self.pfcnet.append(layers)
            
        for i in range(network_count):
            # MDCBC
            layers = nn.ModuleList()
            for j in range(self.gcn_k):
                layers.append(GCNConv(self.unit, h1, improved= True, cached=True))
                layers.append(nn.BatchNorm1d(h1,track_running_stats=False) )
                if self.gcn_k > 1:
                    layers.append(GCNConv(h1, h2, improved= True, cached=True))
                else:
                    layers.append(GCNConv(h1, 2, improved= True, cached=True))
            if self.gcn_k > 1:
                layers.append(nn.Linear(self.gcn_k * h2, int(self.gcn_k * h2 / 2)))
                layers.append(nn.Linear(int(self.gcn_k * h2 / 2), 2))
                
            self.mdcbcnet.append(layers)
        for i in range(network_count):
            # V1C
            layers = nn.ModuleList()
            for j in range(self.gcn_k):
                layers.append(GCNConv(self.unit, h1, improved= True, cached=True))
                layers.append(nn.BatchNorm1d(h1,track_running_stats=False))
                if self.gcn_k > 1:
                    layers.append(GCNConv(h1, h2, improved= True, cached=True))
                else:
                    layers.append(GCNConv(h1, 2, improved= True, cached=True))
            if self.gcn_k > 1:
                layers.append(nn.Linear(self.gcn_k * h2, int(self.gcn_k * h2 / 2)) )
                layers.append(nn.Linear(int(self.gcn_k * h2 / 2), 2) )
                
            self.v1cnet.append(layers)
        for i in range(network_count):
            # SHA
            layers = nn.ModuleList()
            for j in range(self.gcn_k):
                layers.append(GCNConv(self.unit, h1, improved= True, cached=True) )
                layers.append(nn.BatchNorm1d(h1,track_running_stats=False) )
                if self.gcn_k > 1:
                    layers.append(GCNConv(h1, h2, improved= True, cached=True) )
                else:
                    layers.append(GCNConv(h1, 2, improved= True, cached=True) )
            if self.gcn_k > 1:
                layers.append(nn.Linear(self.gcn_k * h2, int(self.gcn_k * h2 / 2)) )
                layers.append(nn.Linear(int(self.gcn_k * h2 / 2), 2) )
                
            self.shanet.append(layers)    
        self.gating = nn.Linear(featsize, network_count * 4).to(devices[0])
        self.gating_weights = self.gating.weight
        
    def forward(self,flatten, features, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights):
        
        # Networks
        pfcconcatlist = []
        mdcbcconcatlist = []
        v1cconcatlist = []
        shaconcatlist = []
        for i in range(network_count):
            pfcconcatlist.append(torch.zeros((data.x.shape[0], self.gcn_k * h2), dtype = torch.float, requires_grad = True) )
            mdcbcconcatlist.append(torch.zeros((data.x.shape[0], self.gcn_k * h2), dtype = torch.float, requires_grad = True) )
            v1cconcatlist.append(torch.zeros((data.x.shape[0], self.gcn_k * h2), dtype = torch.float, requires_grad = True) )
            shaconcatlist.append(torch.zeros((data.x.shape[0], self.gcn_k * h2), dtype = torch.float, requires_grad = True) )
        expert_results = []
        
        for i in range(network_count):
            for j in range(self.gcn_k):
                x = self.pfcnet[i][j * 3](flatten[pfcgpumask[i]], pfcnetworks[i])
                x = F.relu(x)
                x = self.pfcnet[i][j * 3 + 1](x)
                x = F.dropout(x, training=self.training)
                x = self.pfcnet[i][j * 3 + 2](x, pfcnetworks[i])
                if self.gcn_k > 1:
                    pfcconcatlist[i][:,j * h2:j * h2 + h2] = x
                else:
                    expert_results.append(x)
            if self.gcn_k > 1:
                pfcconcatlist[i] = self.pfcnet[i][self.gcn_k * 3](pfcconcatlist[i])
                pfcconcatlist[i] = F.relu(pfcconcatlist[i])
                pfcconcatlist[i] = self.asdpfcnet[i][self.gcn_k * 3 + 1](pfcconcatlist[i])
                expert_results.append(pfcconcatlist[i])
            
            for j in range(self.gcn_k):
                x = self.mdcbcnet[i][j * 3](flatten[mdcbcgpumask[i]], mdcbcnetworks[i])
                x = F.relu(x)
                x = self.mdcbcnet[i][j * 3+ 1](x)
                x = F.dropout(x, training=self.training)
                x = self.mdcbcnet[i][j * 3 + 2](x, mdcbcnetworks[i])
                if self.gcn_k > 1:
                    mdcbcconcatlist[i][:,j * h2: j * h2 + h2] = x
                else:
                    expert_results.append(x)
            if self.gcn_k > 1:
                mdcbcconcatlist[i] = self.mdcbcnet[i][self.gcn_k * 3](mdcbcconcatlist[i])
                mdcbcconcatlist[i] = F.relu(asdmdcbcconcatlist[i])
                mdcbcconcatlist[i] = self.asdmdcbcnet[i][self.gcn_k * 3 + 1](mdcbcconcatlist[i])
                expert_results.append(mdcbcconcatlist[i])
            
            for j in range(self.gcn_k):
                x = self.v1cnet[i][j * 3](flatten[v1cgpumask[i]], v1cnetworks[i])
                x = F.relu(x)
                x = self.v1cnet[i][j * 3 + 1](x)
                x = F.dropout(x, training=self.training)
                x = self.v1cnet[i][j * 3 + 2](x, v1cnetworks[i])
                if self.gcn_k > 1:
                    v1cconcatlist[i][:,j * h2: j * h2 + h2] = x
                else:
                    expert_results.append(x)
            if self.gcn_k > 1:
                v1cconcatlist[i] = self.v1cnet[i][self.gcn_k * 3](v1cconcatlist[i])
                v1cconcatlist[i] = F.relu(v1cconcatlist[i])
                v1cconcatlist[i] = self.v1cnet[i][self.gcn_k * 3 + 1](v1cconcatlist[i])
                expert_results.append(v1cconcatlist[i])
            
            for j in range(self.gcn_k):
                x = self.shanet[i][j * 3](flatten[shagpumask[i]], shanetworks[i])
                x = F.relu(x)
                x = self.shanet[i][j * 3 + 1](x)
                x = F.dropout(x, training=self.training)
                x = self.shanet[i][j * 3 + 2](x, shanetworks[i])
                if self.gcn_k > 1:
                    shaconcatlist[i][:,j * h2: j * h2 + h2] = x
                else:
                    expert_results.append(x)
            if self.gcn_k > 1:
                shaconcatlist[i] = self.shanet[i][self.gcn_k * 3](shaconcatlist[i])
                shaconcatlist[i] = F.relu(shaconcatlist[i])
                shaconcatlist[i] = self.shanet[i][self.gcn_k * 3 + 1](shaconcatlist[i])
                expert_results.append(shaconcatlist[i])      
                
        for i in range(len(expert_results)):
            expert_results[i] = F.log_softmax(expert_results[i], dim = 1)
            expert_results[i] = expert_results[i]

        weights = self.gating(features[0])
        weights = F.softmax(weights, dim = 1)
        
        self.experts = weights
        self.gating_weights = self.gating.weight
        extended = torch.zeros((data.x.shape[0], len(expert_results)), dtype = torch.float, requires_grad = True)
        for i in range(len(expert_results)):
            extended[:, i] = expert_results[i][:,1] 
        
        extended2 = torch.zeros((data.x.shape[0], len(expert_results)), dtype = torch.float, requires_grad = True)

        for i in range(len(expert_results)):
            extended2[:, i] = expert_results[i][:,0]
        
        results = torch.sum(weights * extended, dim = 1)
        results2 = torch.sum(weights * extended2, dim = 1)
        
        final = torch.zeros( (data.x.shape[0], 2), dtype = torch.float, requires_grad = True)
        final[:, 1] = results
        final[:, 0] = results2
        return final
        
class DeepND(torch.nn.Module): # Multi Task DeepND
    def __init__(self):
        super(DeepND, self).__init__()
        self.unit = 15
        torch.set_rng_state(torch.load("/mnt/ilayda/gcn_exp_results/MultiExp"+str(experiment)+"/deepND_experiment_torch_random_state"))
        self.commonmlp = nn.Linear(featsize, self.unit).to(devices[0])
        torch.set_rng_state(torch.load("/mnt/ilayda/gcn_exp_results/MultiExp"+str(experiment)+"/deepND_experiment_torch_random_state"))
        self.ASDBranch= DeepND_ST(featsize=featsizeasd)  
        torch.set_rng_state(torch.load("/mnt/ilayda/gcn_exp_results/MultiExp"+str(experiment)+"/deepND_experiment_torch_random_state"))
        self.IDBranch= DeepND_ST(featsize=featsizeid)        
      
    # data contains a packed structure: Features and first graph's edge indices
    # Currentlt, network is configured to work on a single graph.
    #To enable multiple graphs, uncomment correspoding convolution layers in addition to MLP layer at the end
    def forward(self, features, asdfeatures, idfeatures, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights):
        
        flat = self.commonmlp(features)
        flat = F.leaky_relu(flat, negative_slope=1.5)
        flatten = []
        for i in range(len(devices)):
            flatten.append(flat)
                
        final1 = self.ASDBranch(flatten, asdfeatures, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights)
        final2 = self.IDBranch(flatten, idfeatures, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights)

        return final1, final2
