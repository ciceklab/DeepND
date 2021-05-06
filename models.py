"""
models.py
Multitask and singletask model constructors for DeepND 
Bilkent University, Department of Computer Engineering
Ankara, 2020
"""
import os
import pandas as pd
import numpy as np
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class DeepND_ST(torch.nn.Module): # Single Task DeepND, used as a building block for multitask model
    def __init__(self, devices, gpu_mask, feature_count, hidden_units, network_count, utility_gpu, common_layer_output_size, moe_feature_count=None):
        super(DeepND_ST, self).__init__()
        self.hidden_units = hidden_units
        self.gpu_mask = gpu_mask
        self.feature_count = feature_count
        self.network_count = network_count
        self.utility_gpu = utility_gpu
        self.common_neurons = common_layer_output_size
        self.layers = nn.ModuleList()
        if moe_feature_count:
            self.moe_feature_count = moe_feature_count
        else:
            self.moe_feature_count= feature_count
        for i in range(network_count):
            
            self.layers.append(GCNConv(self.common_neurons, self.hidden_units, improved= True, cached=True).to(devices[gpu_mask[i]]))
            self.layers.append(nn.BatchNorm1d(self.hidden_units,track_running_stats=False).to(devices[gpu_mask[i]]))
            self.layers.append(GCNConv(self.hidden_units, 2, improved= True, cached=True).to(devices[gpu_mask[i]]))           
            #modules.append(layers)
        self.gating = nn.Linear(self.moe_feature_count, self.network_count).to(devices[self.utility_gpu])
        
    def forward(self, features, moe_features, networks, devices, gpu_mask, flatten):
        self.instance_count = features[devices[0]].shape[0]
        # Networks
        self.expert_results = []
        self.expert_weights = None
        self.expert_probabilities = []
        
        instance_count = features[devices[0]].shape[0]
        
        for i in range(self.network_count):
            
            x = self.layers[i * 3 ](flatten[devices[gpu_mask[i]]], networks[i])
            x = F.relu(x)
            x = self.layers[i * 3 + 1](x) 
            #x = F.dropout(x, training=self.training)
            
            x = self.layers[i * 3 + 2](x, networks[i])
            self.expert_results.append(x)

        for i in range(self.network_count):
            self.expert_probabilities.append(F.softmax(self.expert_results[i], dim=1))
            
            self.expert_results[i] = F.log_softmax(self.expert_results[i], dim = 1)
            self.expert_results[i] = self.expert_results[i].to(devices[self.utility_gpu]) 

        weights = self.gating(moe_features[devices[self.utility_gpu]])
        weights = F.softmax(weights, dim = 1)
        self.expert_weights = weights
        
        extended = torch.zeros((self.instance_count, self.network_count), dtype = torch.float, requires_grad = True).to(devices[self.utility_gpu])
        for i in range(len(self.expert_results)):
            extended[:, i] = self.expert_results[i][:,1] 
        results = torch.sum(weights * extended, dim = 1).to(devices[self.utility_gpu])
        
        
        extended2 = torch.zeros((self.instance_count, self.network_count), dtype = torch.float, requires_grad = True).to(devices[self.utility_gpu])
        for i in range(len(self.expert_results)):
            extended2[:, i] = self.expert_results[i][:,0] 
        results2 = torch.sum(weights * extended2, dim = 1).to(devices[self.utility_gpu])
        
        extended_probabilities = torch.zeros((self.instance_count, self.network_count), dtype = torch.float, requires_grad = True).to(devices[self.utility_gpu])
        for i in range(len(self.expert_probabilities)):
            extended_probabilities[:, i] = self.expert_probabilities[i][:,1].to(devices[self.utility_gpu])
            
        extended_probabilities2 = torch.zeros((self.instance_count, self.network_count), dtype = torch.float, requires_grad = True).to(devices[self.utility_gpu])
        for i in range(len(self.expert_probabilities)):
            extended_probabilities2[:, i] = self.expert_probabilities[i][:,0].to(devices[self.utility_gpu])
            
        results_probabilities = torch.sum(weights * extended_probabilities, dim = 1).to(devices[self.utility_gpu])
        results_probabilities2 = torch.sum(weights * extended_probabilities2, dim = 1).to(devices[self.utility_gpu])
        # Since we take log_softmax the results are not probabilities. Therefore, we also keep softmaxed probability versions.
        
        results_concat = torch.zeros((instance_count,2), dtype = torch.float, requires_grad = True).to(devices[self.utility_gpu])
        results_concat[:,0] = results2
        results_concat[:,1] = results
        
        results_probs_concat = torch.zeros((instance_count,2), dtype = torch.float, requires_grad = True).to(devices[self.utility_gpu])
        results_probs_concat[:,0] = results_probabilities2
        results_probs_concat[:,1] = results_probabilities
        
        return [results_concat, results_probs_concat]
        
class DeepND(torch.nn.Module): # Multi Task DeepND
    def __init__(self, devices, gpu_mask, feature_sizes, common_layer_output_size, hidden_units, networks, utility_gpu, root, experiment, disorder_name, moe_input_size):
        super(DeepND, self).__init__()
        torch.set_rng_state(torch.load(root + disorder_name + "Exp" + str(experiment) + "/deepND_experiment_torch_random_state"))
        self.branches = nn.ModuleList()
        if len(feature_sizes) == 1: # There is only 1 feature set, meaning that it is singletask
            self.branches.append(DeepND_ST(devices, gpu_mask, feature_sizes[0], hidden_units, len(networks), utility_gpu, feature_sizes[0], moe_input_size[0]))
        else: # Multitask setting
            self.commonmlp = nn.Linear(feature_sizes[-1], common_layer_output_size).to(devices[utility_gpu])
            
            for i in range(len(feature_sizes) - 1):
                if os.path.isfile(root + "MultiExp" + str(experiment) + "/deepND_experiment_torch_random_state"):
                    torch.set_rng_state(torch.load(root + "MultiExp" + str(experiment) + "/deepND_experiment_torch_random_state"))
                self.branches.append(DeepND_ST(devices, gpu_mask, feature_sizes[i], hidden_units, len(networks), utility_gpu, common_layer_output_size, moe_input_size[i]))
     
      
    def forward(self, features, moe_features, networks, devices, gpu_mask, utility_gpu):
        if (len(features) - 1) != 0:
            flat = self.commonmlp(features[-1][devices[utility_gpu]])
            
            flat = F.leaky_relu(flat, negative_slope=1.5)
            #flat = F.relu(flat)
            flatten = {}
            for i in range(len(devices)):
                flatten[devices[i]] = flat.to(devices[i])
        branch_results = []
        if (len(features) - 1) != 0:
            for i in range (len(features) - 1):
                branch_results.append(self.branches[i](features[i], moe_features[i], networks, devices, gpu_mask, flatten))
        else:
            branch_results.append(self.branches[0](features[0], moe_features[0], networks, devices, gpu_mask, features[0]))

        return branch_results
