"""
utils.py
Utiliy functions for DeepND 
Bilkent University, Department of Computer Engineering
Ankara, 2020
"""

import numpy as np
import pandas as pd
from datetime import timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import csv
import math

def memoryUpdate(usage = 0, cached = 0):
    # Memory Update!
    current_usage = 0
    current_cached = 0
    for d in range(torch.cuda.device_count()):
        current_usage += torch.cuda.max_memory_allocated(device='cuda:'+str(d))
        current_cached += torch.cuda.max_memory_cached(device='cuda:'+str(d))
    usage = max(usage,current_usage)
    cached = max(cached, current_cached)
    print("GPU Memory Usage:", usage / 10**9, "GB Used, ", cached / 10**9, "GB Cached")
    return usage, cached
    
def maskCheck(masks):
    l  = []
    for mask in masks:
        l.append(len(mask))
    l = set(l)
    l = list(l)
    if len(l) > 1:
        raise ValueError('GPU masks and network region has different munbers!')

def weight_reset(m):
    if isinstance(m, GCNConv) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()
    # Xavier initialization for layers
    if isinstance(m, nn.Linear) or isinstance(m, GCNConv):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
        
def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False
        
def unfreeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = True

def intersect_lists(source_list, target_list, lookup):
    #Intersects gene list from multple sources
    source_indices = []
    target_indices = []
    result = []
    not_found_indices = []
    not_found_item = []
    not_found_count = 0
    for source_index, source_item in enumerate(source_list):
        found = 0
        for target_index, target_item in enumerate(target_list):
            if source_item.lower() == target_item.lower():
                source_indices.append(source_index)
                target_indices.append(target_index)
                result.append(target_item.lower())
                found = 1
                break
        if found == 0:
            for target_index, target_item in enumerate(target_list):
                #if source_item.lower() in lookup and target_item.lower() in lookup and does_intersect(lookup[source_item.lower()], lookup[target_item.lower()]): #Dictionary search
                if source_item.lower() in lookup and target_item.lower() in lookup and lookup[source_item.lower()] and  lookup[target_item.lower()] and  lookup[source_item.lower()][-1] ==  lookup[target_item.lower()][-1] : #Dictionary search
                    source_indices.append(source_index)
                    target_indices.append(target_index)
                    result.append(target_item.lower())
                    found = 1
                    print("Found in Dictionary!", source_item , target_item)
                    break
        if found == 0:
            not_found_indices.append(source_index)
            not_found_item.append(source_item)
            not_found_count += 1
            #print("The gene {0} is not found. Not Found Count:{1}".format(source_item, not_found_count))      
    return result, source_indices, not_found_indices, target_indices
            
def does_intersect(source_list, target_list):
    for source_item in source_list:
        for target_item in target_list:
            if source_item.lower() == target_item.lower():
                return True
    return False

def constructGeneDictionary(path):
    # Constructs a dictionary for gene aliases and ids
    genes = dict()
    lineCount = 1
    with open(path) as tsv:
        for line in csv.reader(tsv, dialect = csv.excel_tab, delimiter  = "\t"): #You can also use delimiter="\t" rather than giving a dialect.
            if line[0] == "Approved symbol":
                continue
            for item in line:
                if item == "":
                    continue
                gene_item = item.split(", ")
                #if lineCount == 10282:
                  #print(gene_item)
                for comma_item in gene_item:
                    gene_list = []
                    for item2 in line:
                        if item2 == "":
                            continue
                        gene_item2 = item2.split(", ")
                        for comma_item2 in gene_item2:
                            if comma_item2 == comma_item:
                                continue
                            gene_list.append(comma_item2.lower())                    
                    genes[comma_item.lower()] = gene_list
            lineCount += 1         
    return genes

def load_networks(root, devices,  pfcgpumask, mdcbcgpumask, shagpumask, v1cgpumask, mask = None , network_count =13): 
    periods = ["1-3", "2-4", "3-5", "4-6", "5-7", "6-8", "7-9", "8-10", "9-11", "10-12", "11-13", "12-14", "13-15"]
    if mask:
        pfc08Mask = mask
        mdcbc08Mask = mask
        v1c08Mask = mask
        sha08Mask= mask
    else:
        pfc08Mask = [i for i in range(network_count)]
        mdcbc08Mask = [i for i in range(network_count)]
        v1c08Mask = [i for i in range(network_count)]
        sha08Mask= [i for i in range(network_count)]
    
    pfcnetworks = []
    pfcnetworkweights = []
    mdcbcnetworks = []
    mdcbcnetworkweights = []
    v1cnetworks = []
    v1cnetworkweights = []
    shanetworks = []
    shanetworkweights = []
    
    for period in pfc08Mask:
        pfcnetworks.append(torch.load(root + "Data/EdgeTensors/PFC" + periods[period] + "wTensor.pt").type(torch.LongTensor))
        pfcnetworkweights.append(torch.abs(torch.load(root + "Data/EdgeTensors/PFC" + periods[period] + "EdgeWeightTensor.pt").type(torch.FloatTensor)[0,:]))
    
    for period in mdcbc08Mask:
        mdcbcnetworks.append(torch.load(root + "Data/EdgeTensors/MDCBC" + periods[period] + "wTensor.pt").type(torch.LongTensor))
        mdcbcnetworkweights.append(torch.abs(torch.load(root + "Data/EdgeTensors/MDCBC" + periods[period] + "EdgeWeightTensor.pt").type(torch.FloatTensor)[0,:]))
      
    for period in v1c08Mask:
        v1cnetworks.append(torch.load(root + "Data/EdgeTensors/V1C" + periods[period] + "wTensor.pt").type(torch.LongTensor)) 
        v1cnetworkweights.append(torch.abs(torch.load(root + "Data/EdgeTensors/V1C" + periods[period] + "EdgeWeightTensor.pt").type(torch.FloatTensor)[0,:]))
    
    for period in sha08Mask:
        shanetworks.append(torch.load(root + "Data/EdgeTensors/SHA" + periods[period] + "wTensor.pt").type(torch.LongTensor)) 
        shanetworkweights.append(torch.abs(torch.load(root + "Data/EdgeTensors/SHA" + periods[period] + "EdgeWeightTensor.pt").type(torch.FloatTensor)[0,:]))
    
    for i in range(len(pfc08Mask)):
        pfcnetworks[i] = pfcnetworks[i].to(devices[pfcgpumask[i]])
        pfcnetworkweights[i] = pfcnetworkweights[i].to(devices[pfcgpumask[i]])
        mdcbcnetworks[i] = mdcbcnetworks[i].to(devices[mdcbcgpumask[i]])
        mdcbcnetworkweights[i] = mdcbcnetworkweights[i].to(devices[mdcbcgpumask[i]])
        v1cnetworks[i] = v1cnetworks[i].to(devices[v1cgpumask[i]])
        v1cnetworkweights[i] = v1cnetworkweights[i].to(devices[v1cgpumask[i]])
        shanetworks[i] = shanetworks[i].to(devices[shagpumask[i]])
        shanetworkweights[i] = shanetworkweights[i].to(devices[shagpumask[i]])
        
    return pfcnetworks, pfcnetworkweights, mdcbcnetworks, mdcbcnetworkweights, v1cnetworks, v1cnetworkweights, shanetworks, shanetworkweights

def load_goldstandards(root,  geneNames_all, geneDict, diseasename = "ASD"):
    """GOLD STANDARDS"""
    # Following section loads gold standard genes
    # To use other standards, following section needs to be changed
    
    pos_gold_standards = pd.read_csv(root + "Data/" + diseasename + "_Pos_Gold_Standards.csv")
    neg_gold_standards = pd.read_csv(root + "Data/" + diseasename + "_Neg_Gold_Standards.csv")
    
    pos_gold_std = pos_gold_standards.values
    neg_gold_std = neg_gold_standards.values
    
    pos_gold_std_genes = [str(item) for item in pos_gold_std[:,0]]
    pos_gold_std_evidence = [str(item) for item in pos_gold_std[:,2]]
    neg_gold_std_genes = [str(item) for item in neg_gold_std[:,0]]
    
    y = torch.zeros(len(geneNames_all), dtype = torch.long)
    
    pgold_tada_intersect, pgold_indices, pgold_delete_indices, g_bs_tada_intersect_indices = intersect_lists(pos_gold_std_genes , [str(item) for item in geneNames_all], geneDict)
    ngold_tada_intersect, ngold_indices, ngold_delete_indices, n_bs_tada_intersect_indices = intersect_lists(neg_gold_std_genes , [str(item) for item in geneNames_all], geneDict)
    y[g_bs_tada_intersect_indices] = 1
    y[n_bs_tada_intersect_indices] = 0
    gold_evidence = [pos_gold_std_evidence[item] for item in pgold_indices]
    
    print(len(pgold_tada_intersect), " Many Positive Gold Standard Genes are Found!")
    print(len([pos_gold_std_genes[item] for item in pgold_delete_indices]), " Many Positive Gold Standard Genes Cannot be Found!")
    print(len(ngold_tada_intersect), " Many Negative Gold Standard Genes are Found!")
    print(len([neg_gold_std_genes[item] for item in ngold_delete_indices]), " Many Negative Gold Standard Genes Cannot be Found!")
    pos_neg_intersect, pos_indices, not_found_indices , neg_indices = intersect_lists(pgold_tada_intersect , ngold_tada_intersect, geneDict)
    print("Positive and Negative Gold Standard Gene Intersection List:", pos_neg_intersect)
    print("Positive and Negative Gold Standard Gene Intersection List Length:", len(pos_neg_intersect))
    return  g_bs_tada_intersect_indices, n_bs_tada_intersect_indices, y, pos_gold_std_evidence, gold_evidence

def create_validation_set(g_bs_tada_intersect_indices, n_bs_tada_intersect_indices, gold_evidence, k, state):                                 
    # k for k-fold cross validation
    # If another validation set is used, gene counts must be updated. This part could be done automatically 
    # as well by checking gene evidences and standard values from files
    e1_gene_count = 0
    e2_gene_count = 0
    e3e4_gene_count = 0
    e1_gene_indices = []
    e2_gene_indices = []
    e3e4_gene_indices = []
    pos_gold_standards = []
    neg_gold_standards = []
    for index,i in enumerate(gold_evidence):
        if i == "E1":
            e1_gene_count += 1
            e1_gene_indices.append(g_bs_tada_intersect_indices[index])
        elif i == "E2":
            e2_gene_count += 1
            e2_gene_indices.append(g_bs_tada_intersect_indices[index])
        else:
            e3e4_gene_count += 1
            e3e4_gene_indices.append(g_bs_tada_intersect_indices[index])
    e1_fold_size = math.ceil(e1_gene_count / k)
    e2_fold_size = math.ceil(e2_gene_count / k)
    e3e4_fold_size = math.ceil(e3e4_gene_count / k)
    neg_gene_count = len(n_bs_tada_intersect_indices)
    neg_fold_size = math.ceil(neg_gene_count / k)
    
    print("E1 Gene Count:", e1_gene_count)
    print("E2 Gene Count:", e2_gene_count)
    print("E3E4 Gene Count:", e3e4_gene_count)
    counts = [e1_gene_count, e2_gene_count, e3e4_gene_count, neg_gene_count]
    # Shuffle all genes
    if state:
        np.random.set_state(state)
    e1_perm = np.random.permutation(e1_gene_count)
    e2_perm = np.random.permutation(e2_gene_count)
    e3e4_perm = np.random.permutation(e3e4_gene_count)
    neg_perm = np.random.permutation(neg_gene_count)
    return e1_gene_indices, e1_perm, e2_gene_indices, e2_perm, e3e4_gene_indices, e3e4_perm, neg_perm, counts

def loadFeatures(root, y, geneNames_all, devices, diseasename = "ASD"):
    features = np.load(root + "Data/" + diseasename + "_TADA_Features.npy")
    features = torch.from_numpy(features).float()
    features = (features - torch.mean(features,0)) / (torch.std(features,0))
    
    data = Data(x=features)
    data = data.to(devices[0])
    
    data.y = y.to(devices[0])
    features = []
    for i in range (len(devices)):
        feature = data.x
        features.append(feature.to(devices[i]))
    return data, features

def writePrediction(predictions, g_bs_tada_intersect_indices, n_bs_tada_intersect_indices, path = "", diseasename="ASD", trial = 10, k = 5):
    predictions /= float(trial*k*(k-1))
    predictions[g_bs_tada_intersect_indices + n_bs_tada_intersect_indices] *= float(k)
    fpred = open( path + "/predict_" + lower(diseasename) +".txt","w+")
    fpred.write('Probability,Gene Name,Gene ID,Positive Gold Standard,Negative Gold Standard\n')
    for index,row in enumerate(predictions):
        if str(geneNames_all[index]) in geneDict:
            fpred.write('%s,%s,%d,%d,%d\n' % (str(row.item()), str(geneDict[str(geneNames_all[index])][0]), geneNames_all[index], 1 if str(geneNames_all[index]) in pos_gold_std_genes else 0, 1 if str(geneNames_all[index]) in neg_gold_std_genes else 0   ) )
        else:
            fpred.write('%s,%s,%d,%d,%d\n' % (str(row.item()), str(geneNames_all[index]), geneNames_all[index], 1 if str(geneNames_all[index]) in pos_gold_std_genes else 0, 1 if str(geneNames_all[index]) in neg_gold_std_genes else 0 ) )
    fpred.close()
                                 
def writeExperimentStats( aucs, aupr, path = "", diseasename="ASD", trial = 10, k = 5, init_time = 0.0, network_count =13, mode = 0):
    
    f = open( path + "/runreport.txt","w")
    f.write("Number of networks per region: %d\n" % network_count)
    print("Number of networks per region:" , network_count)
    if not mode:
        f.write("Generated test results, i.e. no training process.")
    f.write("\nDone in %s hh:mm:ss.\n" % timedelta( seconds = (time.time()-init_time) ) )
    if diseasename == "Multi":
        # Multi Task Experiment Stats
        diseases = ["ASD", "ID"]
        for i in range(2):
            f.write("Disease : %s\n" % diseases[i])
            print("Disease :", diseases[i])
            f.write("-"*20+"\n")
            f.write("\nMean (\u03BC) AUC of All Runs:%f\n" % np.mean(aucs[i]) )
            print(" Mean(\u03BC) AUC of All Runs:", np.mean(aucs[i]) )
            f.write(" \u03C3 of AUCs of All Runs:%f\n" % np.std(aucs[i]) )
            print("\u03C3 of AUCs of All Runs:", np.std(aucs[i]) )
            f.write(" Median of AUCs of All Runs:%f\n" % np.median(aucs[i]) )
            print(" Meadian of AUCs of All Runs:", np.median(aucs[i]) )
            f.write("\n Mean (\u03BC) APRC of All Runs:%f\n" % np.mean(aupr[i]) )
            print(" Mean(\u03BC) AUPR of All Runs:", np.mean(aupr[i]) )
            f.write(" \u03C3 of AUPR of All Runs:%f\n" % np.std(aupr[i]) )
            print(" \u03C3 of AUPR of All Runs:", np.std(aupr[i]) )
            f.write(" Median of AUPR of All Runs:%f\n" % np.median(aupr[i]) )
            print("Meadian of AUCs of All Runs:", np.median(aupr[i]) )
        for i in range(2):            
            f.write("*"*80+"\n") 
            for j in range(len(aucs[i])):
                f.write("%s AUC:%f\n" % (diseasename, aucs[i][j]))    
                f.write("-"*20+"\n") 
            for j in range(len(aupr)):
                f.write("%s AUPR:%f\n" % (diseasename , aupr[i][j]))    
            f.write("-"*20+"\n") 
        
    else: 
        # Single Task Experiment Stats
        f.write("Disease : %s\n" % diseasename)
        print("Disease :", diseasename)
        f.write("-"*20+"\n")
        f.write("\nMean (\u03BC) AUC of All Runs:%f\n" % np.mean(aucs) )
        print(" Mean(\u03BC) AUC of All Runs:", np.mean(aucs) )
        f.write(" \u03C3 of AUCs of All Runs:%f\n" % np.std(aucs) )
        print("\u03C3 of AUCs of All Runs:", np.std(aucs) )
        f.write(" Median of AUCs of All Runs:%f\n" % np.median(aucs) )
        print(" Meadian of AUCs of All Runs:", np.median(aucs) )
        f.write("\n Mean (\u03BC) APRC of All Runs:%f\n" % np.mean(aupr) )
        print(" Mean(\u03BC) AUPR of All Runs:", np.mean(aupr) )
        f.write(" \u03C3 of AUPR of All Runs:%f\n" % np.std(aupr) )
        print(" \u03C3 of AUPR of All Runs:", np.std(aupr) )
        f.write(" Median of AUPR of All Runs:%f\n" % np.median(aupr) )
        print("Meadian of AUCs of All Runs:", np.median(aupr) )
                                    
        f.write("*"*80+"\n") 
        for i in range(len(aucs)):
            f.write("%s AUC:%f\n" % (diseasename, aucs[i]))    
        f.write("-"*20+"\n") 
        for i in range(len(aupr)):
            f.write("%s AUPR:%f\n" % (diseasename , aupr[i]))    
        f.write("-"*20+"\n") 
        
    f.close()
    print("Generated results for ", diseasename )
    print("Done in ", timedelta( seconds = (time.time()-init_time) ) , "hh:mm:ss." )
       

