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
import torch.cuda as cutorch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import csv
import math
import time
from scipy.sparse import identity
import subprocess

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
    neg_gene_indices = []
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
    for item in n_bs_tada_intersect_indices:
        neg_gene_indices.append(item)
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
    
    return [e1_gene_indices, e2_gene_indices, e3e4_gene_indices, neg_gene_indices],[e1_perm, e2_perm, e3e4_perm, neg_perm], counts


 
def write_prediction(predictions, e1_indices, e2_indices, e3e4_indices, negative_indices, feature_names, root, task_name, trial, k, experiment, experiment_name):
    fpred = open( root + experiment_name + "Exp" + str(experiment) + "/predict_" + task_name.lower() +".csv","w+")
    
    if feature_names == "indices":
        fpred.write('Index, Probability,Positive Gold Standard,Negative Gold Standard, Evidence Level\n')
        for index,row in enumerate(predictions):
            evidence_level = ""
            if index in e1_indices:
                evidence_level = "E1"
            elif index in e2_indices:
                evidence_level = "E2"
            elif index in e3e4_indices:
                evidence_level = "E3E4"
            elif index in negative_indices:
                evidence_level = "Negative"
            fpred.write('%d,%s,%d,%d,%s\n' % (index, str(row.item()), 1 if (index in e1_indices or index in e2_indices or index in e3e4_indices) else 0, 1 if index in negative_indices else 0, evidence_level) )
    else:
        feature_names_file = pd.read_csv(feature_names.strip())
        headers = feature_names_file.columns.tolist()
        results_header = ""
        format_string = ""
        
        for header in headers:
            results_header += header + ","
            format_string += "%s,"
            
        results_header += "Probability, Positive Gold Standard, Negative Gold Standard, Evidence Level\n"
        format_string += "%s,%d,%d,%s"
        fpred.write(results_header)
        index = 0
        for prediction_row, feature_names_row in zip(predictions, feature_names_file.values):
            line_tuple = ()
            evidence_level = ""
            if index in e1_indices:
                evidence_level = "E1"
            elif index in e2_indices:
                evidence_level = "E2"
            elif index in e3e4_indices:
                evidence_level = "E3E4"
            elif index in negative_indices:
                evidence_level = "Negative"
            for i in range(len(feature_names_row)):
                line_tuple += (feature_names_row[i],)
            fpred.write(format_string % (line_tuple + (str(prediction_row.item()), 1 if index in e1_indices else 0, 1 if index in negative_indices else 0, evidence_level))  )
            fpred.write("\n")
            index += 1
        
    fpred.close()
def write_experiment_stats (root, aucs, aupr, mmcs, experiment_name, trial, k, init_time, network_count, mode, task_names, experiment):
    f = open( root + experiment_name + "Exp" + str(experiment) + "/runreport.txt","w")
    f.write("Number of networks: %d\n" % network_count)
    print("Number of networks:" , network_count)
    if mode:
        f.write("Generated test results, i.e. no training process.")
    f.write("\nDone in %s hh:mm:ss.\n" % timedelta( seconds = (time.time()-init_time) ) )
    
    for task_index in range(len(aucs)):
        f.write("Task Name: %s\n" % task_names[task_index])
        print("Task Name:", task_names[task_index])
        f.write("-"*20+"\n")
        f.write("\nMean (\u03BC) AUC of All Runs:%f\n" % np.mean(aucs[task_index]) )
        print(" Mean(\u03BC) AUC of All Runs:", np.mean(aucs[task_index]) )
        f.write(" \u03C3 of AUCs of All Runs:%f\n" % np.std(aucs[task_index]) )
        print("\u03C3 of AUCs of All Runs:", np.std(aucs[task_index]) )
        f.write(" Median of AUCs of All Runs:%f\n" % np.median(aucs[task_index]) )
        print(" Median of AUCs of All Runs:", np.median(aucs[task_index]) )
        print("-"*25)
        f.write("\n Mean (\u03BC) AUPRCs of All Runs:%f\n" % np.mean(aupr[task_index]) )
        print(" Mean(\u03BC) AUPRCs of All Runs:", np.mean(aupr[task_index]) )
        f.write(" \u03C3 of AUPRCs of All Runs:%f\n" % np.std(aupr[task_index]) )
        print(" \u03C3 of AUPRCs of All Runs:", np.std(aupr[task_index]) )
        f.write(" Median of AUPRCs of All Runs:%f\n" % np.median(aupr[task_index]) )
        print("Median of AUPRCSs of All Runs:", np.median(aupr[task_index]) )
        print("-"*25)
        f.write("\n Mean (\u03BC) MMCs of All Runs:%f\n" % np.mean(mmcs[task_index]) )
        print(" Mean(\u03BC) MMCs of All Runs:", np.mean(mmcs[task_index]) )
        f.write(" \u03C3 of MMCs of All Runs:%f\n" % np.std(mmcs[task_index]) )
        print(" \u03C3 of MMCs of All Runs:", np.std(mmcs[task_index]) )
        f.write(" Median of MMCs of All Runs:%f\n" % np.median(mmcs[task_index]) )
        print("Median of MMCs of All Runs:", np.median(mmcs[task_index]) )
        
        f.write("*"*80+"\n") 
        for j in range(len(aucs[task_index])):
            f.write("%s AUC:%f\n" % (task_names[task_index], aucs[task_index][j]))    
        f.write("-"*20+"\n") 
        for j in range(len(aupr[task_index])):
            f.write("%s AUPR:%f\n" % (task_names[task_index] , aupr[task_index][j]))    
        f.write("-"*20+"\n") 
    
       
def create_network_list(networks):
    network_files = []
    splitted_tokens = networks.split(",")
    test_network_path = "/mnt/oguzhan/oguzhan_workspace/EdgeTensors/PointEight/" #NEO
    #test_network_path = "/media/hdd1/oguzhan/oguzhan_workspace/EdgeTensors/PointEight/" # GPU2
    regions = ["PFC","MDCBC","V1C","SHA"]
    periods = ["1-3","2-4","3-5","4-6","5-7","6-8","7-9","8-10","9-11","10-12","11-13","12-14","13-15"]
    for token in splitted_tokens:
        if token.strip() == "brainspan_all":
            for region in regions:
                for period in periods:
                    network_files.append(test_network_path + region + period + "wTensor.pt")
        elif token.strip() == "brainspan_no_overlap":
            for region in regions:
                for period in ["1-3", "4-6", "7-9", "10-12", "13-15"]:
                    network_files.append(test_network_path + region + period + "wTensor.pt")
        else:
            if token.strip().split("-")[0].isnumeric() and token.strip().split("-")[1].isnumeric():
                regions = ["PFC","MDCBC","V1C","SHA"]
                for region in regions:
                    network_files.append(test_network_path + region + token.strip() + "wTensor.pt")
            elif "PFC" in token.strip() or "MDCBC" in token.strip() or "V1C" in token.strip() or "SHA" in token.strip():
                network_files.append(test_network_path + token.strip() + "wTensor.pt")
            else:
                network_files.append(token.strip())
    
    networks = []
    for network in network_files:
        networks.append(torch.load(network).type(torch.LongTensor))
        
    return networks
    
def create_feature_set_list(feature_set):
    feature_files = []
    splitted_tokens = feature_set.split(",")
    for token in splitted_tokens:
        if token.strip() == "ASD":
            feature_files.append("Data/ASD_TADA_Features.npy")
        elif token.strip() == "ID":
            feature_files.append("Data/ID_TADA_Features.npy")
        elif token.strip() == "SCZ":
            feature_files.append("Data/SCZ_TADA_Features.npy")
        elif token.strip() == "EPI":
            feature_files.append("Data/EPI_TADA_Features.npy")
        elif token.strip() == "ASDID": # asd & id
            feature_files.append("Data/Multi_TADA_Features.npy")
        elif token.strip() == "ALL":
            feature_files.append("Data/ALL_TADA_Features.npy")
        elif token.strip() == "EPISCZ": # epilepsy & schzoprenia
            feature_files.append("Data/Multi_EPISCZ_TADA_Features.npy")
        elif token.strip() == "ASDSCZ": # asd & schzoprenia
            feature_files.append("Data/Multi_ASDSCZ_TADA_Features.npy")
        elif token.strip() == "ASDEPI": # asd & epilepsy
            feature_files.append("Data/Multi_ASDEPI_TADA_Features.npy")
        elif token.strip() == "IDSCZ": # id & schzoprenia
            feature_files.append("Data/Multi_IDSCZ_TADA_Features.npy")
        elif token.strip() == "IDEPI": # id & epilepsy
            feature_files.append("Data/Multi_IDEPI_TADA_Features.npy")
        elif token.strip() == "Krishnan":
            feature_files.append("/mnt/oguzhan/oguzhan_workspace/Krishnan/KrishnanFeatures.pt")
        else:
            feature_files.append(token.strip())
        
    return feature_files
    
def create_gt_list(root, positive_gt, negative_gt, verbose, k, state, instance_count):
    gt_files = []
    splitted_tokens = positive_gt.split(",")
    neg_splitted_tokens = negative_gt.split(",")
    all_gt_gene_indices = []
    all_gt_gene_permutations = []
    all_gt_gene_counts = []
    all_gt_labels = []
    for index,token in enumerate(splitted_tokens):
        if token.strip() == "ASD":
            gene_indices,gene_permutations, gene_counts, y = __match_ground_truths__(root, "ASD", verbose, k, state)
        elif token.strip() == "ID":
            gene_indices,gene_permutations, gene_counts, y = __match_ground_truths__(root, "ID", verbose, k, state)
        elif token.strip() == "SCZ":
            gene_indices,gene_permutations, gene_counts, y = __match_ground_truths__(root, "SCZ", verbose, k, state)
        elif token.strip() == "EPI":
            gene_indices,gene_permutations, gene_counts, y = __match_ground_truths__(root, "EPI", verbose, k, state)
        elif token.strip() == "ASD_SFARI_E1E2":
            gene_indices,gene_permutations, gene_counts, y = __match_ground_truths__(root, "ASD_SFARI_E1E2", verbose, k, state)
        elif token.strip() == "ASD_SFARI_E1E2E3":
            gene_indices,gene_permutations, gene_counts, y = __match_ground_truths__(root, "ASD_SFARI_E1E2E3", verbose, k, state)
        elif token.strip() == "SFARI_Brueggeman":
            gene_indices,gene_permutations, gene_counts, y = __match_ground_truths__(root, "SFARI_Brueggeman", verbose, k, state)
        elif token.strip() == "SPARK_Pilot":
            gene_indices,gene_permutations, gene_counts, y = __match_ground_truths__(root, "Data/SPARK_Pilot_Pos_Gold_Standards.csv", verbose, k, state,take_path = True,neg_file = "Data/ASD_SPARC_Neg_Gold_Standards.csv")
        else:
            gene_indices,gene_permutations, gene_counts, y = __match_ground_truths__(root, token.strip(), verbose, k, state,take_path = True,neg_file = neg_splitted_tokens[index])
            '''
            negative_gt_file = pd.read_csv(negative_gt.split(",")[index].strip()).values
            positive_gt_file = pd.read_csv(token.strip()).values
            if positive_gt_file.shape[1] == 2: # With evidence weights
                positive_indices = np.array(positive_gt_file[:,0], dtype = np.long)
                positive_evidences = positive_gt_file[:,1]
            elif positive_gt_file.shape[1] == 1: #Without evidence weights
                positive_indices = np.array(positive_gt_file[:,0], dtype = np.long)
                positive_evidences = np.full(positive_indices.shape, "E1")
            else:
                print("Ground truth file ", token.strip()," is expected to have 1 or 2 columns. Please check github.com/ciceklab/DeepND for details.")
                exit(0)
             
            if negative_gt_file.shape[1] == 2: #With evidence weights
                negative_indices = np.array(negative_gt_file[:,0], dtype = np.long)
                negative_evidences = negative_gt_file[:,1]
            elif negative_gt_file.shape[1] == 1: #Without evidence weights
                negative_indices = np.array(negative_gt_file[:,0], dtype = np.long)
                
                negative_evidences = np.full(negative_indices.shape, "E1") #NOT SUPPORTED CURRENTLY !!
            else:
                print("Ground truth file ", token.strip(), " is expected to have 1 or 2 columns. Please check github.com/ciceklab/DeepND for details.")
                exit(0)
            gene_indices, gene_permutations, gene_counts = create_validation_set(positive_indices, negative_indices, positive_evidences, k, state)
            y = torch.zeros((instance_count,), dtype = torch.long)
            y[positive_indices] = 1
            y[negative_indices] = 0
            '''
        all_gt_gene_indices.append(gene_indices)
        all_gt_gene_permutations.append(gene_permutations)
        all_gt_gene_counts.append(gene_counts)
        all_gt_labels.append(y)
    return all_gt_gene_indices, all_gt_gene_permutations, all_gt_gene_counts, all_gt_labels
            
# Private method for gene matching. Written for ASD and ID.

def __match_ground_truths__(root,disorder_name, verbose, k, state, take_path = False, neg_file = ""):
    geneDict = constructGeneDictionary(root + "Data/hugogenes_entrez.txt")
    geneNames_all = pd.read_csv(root + "Data/row-genes.txt", header = None)
    geneNames_all = geneNames_all[0].tolist()
    if take_path == False:
        pos_gold_standards = pd.read_csv( root + "Data/" + disorder_name + "_Pos_Gold_Standards.csv",na_filter=False,verbose=verbose)
    else:
        pos_gold_standards = pd.read_csv( root + disorder_name,na_filter=False,verbose=verbose)
    
    pos_gold_std = pos_gold_standards.values
    print(pos_gold_std[0:10,:])
    pos_gold_std_genes = [str(item) for item in pos_gold_std[:,0]]
    pos_gold_std_evidence = [str(item) for item in pos_gold_std[:,2]]
    pgold_tada_intersect, pgold_indices, pgold_delete_indices, g_bs_tada_intersect_indices = intersect_lists(pos_gold_std_genes , [str(item) for item in geneNames_all], geneDict)
    gold_evidence = [pos_gold_std_evidence[item] for item in pgold_indices]
    
    if take_path == False:
        neg_gold_standards = pd.read_csv(root + "Data/" + disorder_name + "_Neg_Gold_Standards.csv")
    else:
        neg_gold_standards = pd.read_csv(root + neg_file.strip())
    neg_gold_std = neg_gold_standards.values
    neg_gold_std_genes = [str(item) for item in neg_gold_std[:,0]]
    
    
    pgold_tada_intersect, pgold_indices, pgold_delete_indices, g_bs_tada_intersect_indices = intersect_lists(pos_gold_std_genes , [str(item) for item in geneNames_all], geneDict)
    ngold_tada_intersect, ngold_indices, ngold_delete_indices, n_bs_tada_intersect_indices = intersect_lists(neg_gold_std_genes , [str(item) for item in geneNames_all], geneDict)
    
    pos_neg_intersect, pos_indices, not_found_indices , neg_indices = intersect_lists(pgold_tada_intersect , ngold_tada_intersect, geneDict)
    pos_neg_intersect, pos_indices, not_found_indices , neg_indices = intersect_lists(pgold_tada_intersect , ngold_tada_intersect, geneDict)
    y = torch.zeros(len(geneNames_all), dtype = torch.long)
    y[n_bs_tada_intersect_indices] = 0
    y[g_bs_tada_intersect_indices] = 1
    
    
    if verbose:
        print(len(pgold_tada_intersect), " positive gold standard genes are found for ", disorder_name)
        print(len([pos_gold_std_genes[item] for item in pgold_delete_indices]), " positive gold standard genes cannot be found for ", disorder_name)
        print(len(ngold_tada_intersect), " negative gold standard genes are found for ", disorder_name)
        print(len([neg_gold_std_genes[item] for item in ngold_delete_indices]), " negative gold standard genes cannot be found for ", disorder_name)
        print("Positive and negative gold standard gene intersection list:", pos_neg_intersect)
        print("Positive and negative gold standard gene intersection list length:", len(pos_neg_intersect))
    
    gene_indices, gene_permutations, gene_counts = create_validation_set(g_bs_tada_intersect_indices, n_bs_tada_intersect_indices, gold_evidence, k, state)
    return gene_indices, gene_permutations, gene_counts, y
    
def load_all_features(feature_list):
    features = []
    for feature_set in feature_list:
        feature = None
        file_tokens = feature_set.split(".")
        print(file_tokens)
        if len(file_tokens) == 1 and file_tokens[0].strip().split("_")[0].strip() == "identity":
            i = torch.zeros((2,int(file_tokens[0].strip().split("_")[1].strip())),dtype= torch.long)
            v = torch.zeros(int(file_tokens[0].strip().split("_")[1].strip()),dtype= torch.float)
            for index in range(int(file_tokens[0].strip().split("_")[1].strip())):
                i[0,index] = index
                i[1,index] = index
                v[index] = 1
            #feature = identity(int(file_tokens[0].strip().split("_")[1].strip()), dtype='long', format='dia')
            feature = torch.sparse.LongTensor(i,v,torch.Size([int(file_tokens[0].strip().split("_")[1].strip()),int(file_tokens[0].strip().split("_")[1].strip())]))
            #feature = torch.eye(int(file_tokens[0].strip().split("_")[1].strip()), dtype = torch.long)
        elif file_tokens[1].strip() == "csv" or file_tokens[1].strip() == "txt":
            feature = pd.csv_read(feature_set).values
            feature = torch.from_numpy(feature).float()
            feature = (feature - torch.mean(feature,0)) / (torch.std(feature,0))
        elif file_tokens[1].strip() == "npy":
            feature = np.load(feature_set)
            if feature_set == "Data/ASD_TADA_Features.npy":
                feature = feature[:,[9,10,11,12]]
            elif feature_set == "Data/ID_TADA_Features.npy":
                feature = feature[:,[2,3,7]]
            elif feature_set == "Data/Multi_TADA_Features.npy":
                feature =  feature[:,[9,10,11,12,19,20]]
            print("Feature Shape: ",feature.shape)
            feature = torch.from_numpy(feature).float()
            feature = (feature - torch.mean(feature,0)) / (torch.std(feature,0))
        elif file_tokens[1].strip() == "pt":
            feature = torch.load(feature_set)
            feature = (feature - torch.mean(feature,0)) / (torch.std(feature,0))
        else:
            print("Unsupported extension for feature file at:", feature_set)
            exit(0)
        
        features.append(feature)
    return features
    
def assign_networks(networks, system_gpu_mask, verbose):
    system_gpus = system_gpu_mask.split(",")
    usable_gpus = []
    for gpu in system_gpus:
        usable_gpus.append(int(gpu.strip()))
    memory_usage = get_gpu_memory_map()
    network_weights = []
    edge_sum = 0
    for network in networks:
        edge_sum += network.shape[1]
        
    for network in networks:
        network_weights.append(network.shape[1] / edge_sum)
    
    gpu_weights = []
    gpu_space_sum = 0
    for gpu in memory_usage.keys():
        if gpu in usable_gpus:
            gpu_space_sum += memory_usage[gpu]
        
    for gpu in memory_usage.keys():
        if gpu in usable_gpus:
            gpu_weights.append(gpu_space_sum  / memory_usage[gpu] )
    weight_sum = np.sum(gpu_weights)
    for index,gpu in enumerate(usable_gpus):
        gpu_weights[index] = gpu_weights[index] / weight_sum
    gpu_loads = []
    gpu_masks = []
    for i in range(len(usable_gpus)):
        gpu_loads.append(0.0)
        
    for i in range(len(networks)):
        minIndex = -1
        minLoad = 10000000
        for j,load in enumerate(gpu_loads):
            if load < minLoad:
                minIndex = j
                minLoad = load
        gpu_loads[minIndex] += gpu_weights[minIndex] * network_weights[i]
        gpu_masks.append(minIndex)
    if verbose:
        print("Network GPU assignment:", gpu_masks)
    minIndex = -1
    minLoad = 10000000
    for j,load in enumerate(gpu_loads):
        if load < minLoad:
            minIndex = j
            minLoad = load
    return gpu_masks, minIndex
    


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split(b'\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

 
