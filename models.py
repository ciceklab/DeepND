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
from torch_geometric.data import Data

class DeepND_ST(torch.nn.Module): # Single Task DeepND
    def __init__(self, devices, pfcgpumask, mdcbcgpumask, shagpumask, v1cgpumask, featsize=20, unit=15, genesize = 25825):
        super(DeepND_ST, self).__init__()
        self.unit = unit
        self.h1 = 4
        self.genes = genesize
        self.pfcnet = nn.ModuleList()
        self.mdcbcnet = nn.ModuleList()
        self.v1cnet = nn.ModuleList()
        self.shanet = nn.ModuleList()
        network_count = len(pfcgpumask)    
        for i in range(network_count):
            # PFC
            layers = nn.ModuleList()
            layers.append(GCNConv(self.unit, self.h1, improved= True, cached=True).to(devices[pfcgpumask[i]]))
            layers.append(nn.BatchNorm1d(self.h1,track_running_stats=False).to(devices[pfcgpumask[i]]))
            layers.append(GCNConv(self.h1, 2, improved= True, cached=True).to(devices[pfcgpumask[i]]))           
            self.pfcnet.append(layers)
            
        for i in range(network_count):
            # MDCBC
            layers = nn.ModuleList()
            layers.append(GCNConv(self.unit, self.h1, improved= True, cached=True).to(devices[mdcbcgpumask[i]]))
            layers.append(nn.BatchNorm1d(self.h1,track_running_stats=False).to(devices[mdcbcgpumask[i]]))
            layers.append(GCNConv(self.h1, 2, improved= True, cached=True).to(devices[mdcbcgpumask[i]]))
            self.mdcbcnet.append(layers)
            
        for i in range(network_count):
            # V1C
            layers = nn.ModuleList()            
            layers.append(GCNConv(self.unit, self.h1, improved= True, cached=True).to(devices[v1cgpumask[i]]))
            layers.append(nn.BatchNorm1d(self.h1,track_running_stats=False).to(devices[v1cgpumask[i]]))
            layers.append(GCNConv(self.h1, 2, improved= True, cached=True).to(devices[v1cgpumask[i]]))
            self.v1cnet.append(layers)
            
        for i in range(network_count):
            # SHA
            layers = nn.ModuleList()
            layers.append(GCNConv(self.unit, self.h1, improved= True, cached=True).to(devices[shagpumask[i]]))
            layers.append(nn.BatchNorm1d(self.h1,track_running_stats=False).to(devices[shagpumask[i]]))
            layers.append(GCNConv(self.h1, 2, improved= True, cached=True).to(devices[shagpumask[i]]))
            self.shanet.append(layers)    
        
        self.gating = nn.Linear(featsize, network_count * 4).to(devices[0])
        self.gating_weights = self.gating.weight
        
    def forward(self,flatten, features, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights, devices, pfcgpumask, mdcbcgpumask, v1cgpumask, shagpumask):
        
        # Networks
        pfcconcatlist = []
        mdcbcconcatlist = []
        v1cconcatlist = []
        shaconcatlist = []
        network_count = len(pfcnetworks)
        for i in range(network_count):
            pfcconcatlist.append(torch.zeros((self.genes, 2), dtype = torch.float, requires_grad = True).to(devices[pfcgpumask[i]]))
            mdcbcconcatlist.append(torch.zeros((self.genes, 2), dtype = torch.float, requires_grad = True).to(devices[mdcbcgpumask[i]]))
            v1cconcatlist.append(torch.zeros((self.genes, 2), dtype = torch.float, requires_grad = True).to(devices[v1cgpumask[i]]))
            shaconcatlist.append(torch.zeros((self.genes, 2), dtype = torch.float, requires_grad = True).to(devices[shagpumask[i]]))
        expert_results = []
        
        for i in range(network_count):
            
            x = self.pfcnet[i][0 * 3](flatten[pfcgpumask[i]], pfcnetworks[i])
            x = F.relu(x)
            x = self.pfcnet[i][0 * 3 + 1](x)
            x = F.dropout(x, training=self.training)
            x = self.pfcnet[i][0 * 3 + 2](x, pfcnetworks[i])
            expert_results.append(x)

            x = self.mdcbcnet[i][0 * 3](flatten[mdcbcgpumask[i]], mdcbcnetworks[i])
            x = F.relu(x)
            x = self.mdcbcnet[i][0 * 3+ 1](x)
            x = F.dropout(x, training=self.training)
            x = self.mdcbcnet[i][0 * 3 + 2](x, mdcbcnetworks[i])
            expert_results.append(x)

            x = self.v1cnet[i][0 * 3](flatten[v1cgpumask[i]], v1cnetworks[i])
            x = F.relu(x)
            x = self.v1cnet[i][0 * 3 + 1](x)
            x = F.dropout(x, training=self.training)
            x = self.v1cnet[i][0 * 3 + 2](x, v1cnetworks[i])
            expert_results.append(x)

            x = self.shanet[i][0 * 3](flatten[shagpumask[i]], shanetworks[i])
            x = F.relu(x)
            x = self.shanet[i][0 * 3 + 1](x)
            x = F.dropout(x, training=self.training)
            x = self.shanet[i][0 * 3 + 2](x, shanetworks[i])
            expert_results.append(x)

        for i in range(len(expert_results)):
            expert_results[i] = F.log_softmax(expert_results[i], dim = 1)
            expert_results[i] = expert_results[i].to(devices[0]) 

        weights = self.gating(features[0])
        weights = F.softmax(weights, dim = 1)
        
        self.experts = weights
        extended = torch.zeros((self.genes, len(expert_results)), dtype = torch.float, requires_grad = True).to(devices[0])
        for i in range(len(expert_results)):
            extended[:, i] = expert_results[i][:,1] 
        
        extended2 = torch.zeros((self.genes, len(expert_results)), dtype = torch.float, requires_grad = True).to(devices[0])

        for i in range(len(expert_results)):
            extended2[:, i] = expert_results[i][:,0]
        
        results = torch.sum(weights * extended, dim = 1)
        results2 = torch.sum(weights * extended2, dim = 1)
        
        final = torch.zeros((self.genes, 2), dtype = torch.float, requires_grad = True).to(devices[0])
        final[:, 1] = results
        final[:, 0] = results2
        return final
        
        
class DeepND(torch.nn.Module): # Multi Task DeepND
    def __init__(self):
        super(DeepND, self).__init__()
        self.unit = 15
        torch.set_rng_state(torch.load(root + diseasename + "Exp" + str(experiment) + "/deepND_experiment_torch_random_state"))
        self.commonmlp = nn.Linear(featsize, self.unit).to(devices[0])
        torch.set_rng_state(torch.load(root + diseasename + "Exp" + str(experiment) + "/deepND_experiment_torch_random_state"))
        self.ASDBranch= DeepND_ST(featsize=featsizeasd)  
        torch.set_rng_state(torch.load(root + diseasename + "Exp" + str(experiment) + "/deepND_experiment_torch_random_state"))
        self.IDBranch= DeepND_ST(featsize=featsizeid)        
      
    # data contains a packed structure: Features and first graph's edge indices
    # Currentlt, network is configured to work on a single graph.
    #To enable multiple graphs, uncomment correspoding convolution layers in addition to MLP layer at the end
    def forward(self, features, asdfeatures, idfeatures, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights):
        
        flat = self.commonmlp(features)
        flat = F.leaky_relu(flat, negative_slope=1.5)
        flatten = []
        for i in range(len(devices)):
            flatten.append(flat.to(devices[i]))
                
        final1 = self.ASDBranch(flatten, asdfeatures, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights)
        final2 = self.IDBranch(flatten, idfeatures, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights)

        return final1, final2
