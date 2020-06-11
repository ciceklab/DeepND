"""
deepnd_st.py
Training and testing processes for DeepND ST 
Bilkent University, Department of Computer Engineering
Ankara, 2020
"""
import numpy as np
import pandas as pd
import csv
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.autograd import Variable
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import mannwhitneyu

from models  import *
from utils import *

def deepnd_st(root, path, input_size, mode, l_rate, trial, k, diseasename , devices, pfcgpumask, mdcbcgpumask, shagpumask, v1cgpumask, state,experiment):
    network_count = len(pfcgpumask)
    geneNames_all = pd.read_csv(root + "Data/row-genes.txt", header = None)
    geneNames_all = geneNames_all[0].tolist()
    geneDict = constructGeneDictionary(root + "Data/hugogenes_entrez.txt")
    gene_names_list = [str(item) for item in geneNames_all]
    
    #GOLD STANDARDS
    # Following section loads gold standard genes
    # To use other standards, following section needs to be changed
    
    if diseasename == "ID":
        # ID Validation
        g_bs_tada_intersect_indices, n_bs_tada_intersect_indices, y, pos_gold_std_evidence, gold_evidence = load_goldstandards(root, geneNames_all, geneDict, diseasename = "ID")
    else:
        # ASD Validation
        g_bs_tada_intersect_indices, n_bs_tada_intersect_indices, y, pos_gold_std_evidence, gold_evidence = load_goldstandards(root, geneNames_all, geneDict, diseasename = "ASD")

    # VALIDATION SETS
    e1_gene_indices, e1_perm, e2_gene_indices, e2_perm, e3e4_gene_indices, e3e4_perm, neg_perm, counts = create_validation_set( g_bs_tada_intersect_indices, n_bs_tada_intersect_indices, gold_evidence, k = 5, state = state)
    
    # FEATURES
    if diseasename == "ID":
        data, features = loadFeatures(root, y, geneNames_all, devices, diseasename = "ID")
    else:
        data, features = loadFeatures(root, y, geneNames_all, devices, diseasename = "ASD")
        
    # NETWORKS
    pfcnetworks, pfcnetworkweights, mdcbcnetworks, mdcbcnetworkweights, v1cnetworks, v1cnetworkweights, shanetworks, shanetworkweights = load_networks(root, devices,  pfcgpumask, mdcbcgpumask, shagpumask, v1cgpumask, mask=[11]) 
    
    # MODEL CONSTRUCTION
    model = DeepND_ST(devices, pfcgpumask, mdcbcgpumask, shagpumask, v1cgpumask, featsize=input_size, unit=input_size)
    
    average_att = []
    stddev_att = []
    average_att_gold = []
    stddev_att_gold = []
    average_att_gold_e1 = []
    average_att_gold_e1e2 = []
    average_att_gold_neg = []
    all_att = []
    pre_att = []
    
    for i in range(network_count * 4):
        average_att.append(0.0)
        average_att_gold.append(0.0)
        stddev_att.append(0.0)
        stddev_att_gold.append(0.0)
        average_att_gold_e1.append(0.0)
        average_att_gold_e1e2.append(0.0)
        average_att_gold_neg.append(0.0)
        all_att.append(0.0)
        pre_att.append(0.0)
    
    aucs = []
    aupr = []
    UsandPs = []
    predictions = torch.zeros((len(geneNames_all),1), dtype = torch.float)      
    usage, cached = memoryUpdate()
    # Early Stop Configuration
    early_stop_enabled = True
    old_loss = 100
    early_stop_window = 7
    epoch_count = []       
    for j in range(trial): # 10 here means Run count. Run given times and calculate average AUC found from each run.
        print("Trial:", j+1)
        
        # Losses
        tloss=[]
        vloss=[]
        
        fpr = dict()
        tpr = dict()
        
        usege, cached =  memoryUpdate(usage, cached)
        for k1 in range(k):
        
            e1mask = [e1_gene_indices[index] for index in e1_perm[k1 * math.ceil(counts[0]/k): min(counts[0], (k1 + 1) * math.ceil(counts[0]/k)) ] ]
            data.e1mask = e1mask.copy()
            negmask =  [n_bs_tada_intersect_indices[item] for item in neg_perm[k1 * math.ceil(counts[3]/k) : min(counts[3] , (k1 + 1) * math.ceil(counts[3]/k))] ]
            data.negmask = negmask.copy()
            
            test_mask = [e1_gene_indices[index] for index in e1_perm[k1 * math.ceil(counts[0]/k): min(counts[0], (k1 + 1) * math.ceil(counts[0]/k)) ] ]
            test_mask +=  [n_bs_tada_intersect_indices[item] for item in neg_perm[k1 * math.ceil(counts[3]/k) : min(counts[3] , (k1 + 1) * math.ceil(counts[3]/k))] ]
            data.test_mask = test_mask.copy()
            
            print("Test Mask Length After E1:", len(test_mask))
            print('Test Gene(s):', [gene_names_list[i] for i in test_mask])
            
            test_mask += [e2_gene_indices[index] for index in e2_perm[(k1) * math.ceil(counts[1]/k): min(counts[1], (k1 + 1) * math.ceil(counts[1]/k)) ] ] 
            test_mask += [e3e4_gene_indices[index] for index in e3e4_perm[(k1) * math.ceil(counts[2]/k): min(counts[2], (k1 + 1) * math.ceil(counts[2]/k)) ] ] 
            
            k_e1_perm = np.delete(e1_perm,np.s_[k1*math.ceil(counts[0]/k):min(counts[0],(k1 + 1) * math.ceil(counts[0]/k))],axis=0)
            k_neg_perm = np.delete(neg_perm,np.s_[k1 * math.ceil(counts[3]/k): min(counts[3], (k1 + 1) * math.ceil(counts[3]/k)) ],axis=0)
            k_e2_perm = np.delete(e2_perm,np.s_[k1 * math.ceil(counts[1]/k): min(counts[1], (k1 + 1) * math.ceil(counts[1]/k)) ],axis=0)
            k_e3e4_perm = np.delete(e3e4_perm,np.s_[k1 * math.ceil(counts[2]/k): min(counts[2], (k1 + 1) * math.ceil(counts[2]/k)) ],axis=0)
            
            for k2 in range(k-1): # K-FOLD Cross Validation
                print("Fold", k1+1, "_",  k2+1, "of Trial", j+1)
            
                validation_mask = [e1_gene_indices[index] for index in k_e1_perm[k2 * math.ceil(counts[0]/k): min(counts[0], (k2 + 1) * math.ceil(counts[0]/k)) ] ]
    
                # Add negative genes to validation mask
                validation_mask +=  [n_bs_tada_intersect_indices[item] for item in k_neg_perm[k2 * math.ceil(counts[3]/k) : min(counts[3] , (k2 + 1) * math.ceil(counts[3]/k))] ]
                data.auc_mask = validation_mask.copy()
                
                print('Validation Gene(s):', [gene_names_list[i] for i in validation_mask])
                validation_mask += [e2_gene_indices[index] for index in k_e2_perm[(i) * math.ceil(counts[1]/k): min(counts[1], (i + 1) * math.ceil(counts[1]/k)) ] ] 
                validation_mask += [e3e4_gene_indices[index] for index in k_e3e4_perm[(i) * math.ceil(counts[2]/k): min(counts[2], (i + 1) * math.ceil(counts[2]/k)) ] ] 
            
                # Construct Train Mask
                train_mask = g_bs_tada_intersect_indices + n_bs_tada_intersect_indices
                print("Total Gene Count:", len(train_mask))
                train_mask = [item for item in train_mask if item not in sorted(validation_mask + test_mask)]
    
                print("Final Validation Mask Length:", len(validation_mask))
                print("Final AUC Mask Length:", len(data.auc_mask))
                print("Final Train Mask Length:", len(train_mask))
                print("AUC Mask Length:", len(data.auc_mask))
                
                #Uncomment line below if you want to use sample weights
                sample_weights = torch.ones((len(train_mask)), dtype = torch.float).to(devices[0])
                
                # Uncomment the loop below if you want to use sample weights
                for index, value in enumerate(train_mask):
                    if value in g_bs_tada_intersect_indices:
                        index2 = g_bs_tada_intersect_indices.index(value)
                        evidence = pos_gold_std_evidence[index2]
                        if evidence == "E2":
                            sample_weights[index] = 0.5
                        elif evidence == "E3" or evidence == "E4":
                            sample_weights[index] = 0.25
                    
                data.train_mask = torch.tensor(train_mask, dtype= torch.long)
                data.validation_mask = torch.tensor(validation_mask, dtype=torch.long)
                if mode:
                    #Test Mode
                    model.load_state_dict(torch.load(root + diseasename + "Exp" + str(experiment) + "/deepND_ST_"+diseasename+"_trial"+str(j+1)+"_fold"+str(k1+1)+"_"+str(k2+1)+".pth"))
                    model = model.eval()
                    with torch.no_grad():
                        out = model(features, features, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights, devices, pfcgpumask, mdcbcgpumask, v1cgpumask, shagpumask)
                        _, pred = out.max(dim=1)
                        correct = pred[data.auc_mask].eq(data.y[data.auc_mask]).sum().item()
                        correctTrain = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
                        acc = correct / len(data.auc_mask)
                        accTrain = correctTrain / len(data.train_mask)
                        valLoss = (F.nll_loss(out[data.auc_mask], data.y[data.auc_mask])).to(devices[0])
                        vloss.append(valLoss.cpu().item())
                else:
                    model.apply(weight_reset)
                    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate, weight_decay=0.0001)
                    for epoch in range(1000):
                        model = model.train()
                        optimizer.zero_grad()
                        out = model(features, features, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights, devices, pfcgpumask, mdcbcgpumask, v1cgpumask, shagpumask)
                        loss = (F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight = torch.FloatTensor([1.0, 1.0]).to(devices[0]))).to(devices[0]) # You can adjust class weights using values in FloatTensor
    
                        #Uncomment section below and comment out 3 lines above to enable sample weights.
                        loss = loss * sample_weights
                        loss.mean().backward()
                        tloss.append(loss.mean().cpu().item())
                        optimizer.step()
        
                        model = model.eval()
                        with torch.no_grad():
                            out = model(features, features, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights, devices, pfcgpumask, mdcbcgpumask, v1cgpumask, shagpumask)
                            _, pred = out.max(dim=1)
                            correct = pred[data.auc_mask].eq(data.y[data.auc_mask]).sum().item()
                            correctTrain = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
                            acc = correct / len(data.auc_mask)
                            accTrain = correctTrain / len(data.train_mask)
                            valLoss = (F.nll_loss(out[data.auc_mask], data.y[data.auc_mask])).to(devices[0])
                            vloss.append(valLoss.cpu().item())
                
                        if epoch != 0 and epoch % 25 == 0:
                            print('Validation Accuracy: {:.4f}, Validation Loss: {:.4f} Train Accuracy: {:.4f} Train Loss: {:.4f}'.format(acc,valLoss, accTrain, loss.mean().item()))
                    
                        # Early Stop Checks
                        if early_stop_enabled:
                            if valLoss < old_loss:
                                early_stop_count = 0
                                old_loss = valLoss
                            else:
                                early_stop_count += 1
                                if early_stop_count == early_stop_window:
                                    print("Epoch:", epoch, ", Loss:",loss.mean().item())
                                    break
            
                    torch.save(model.state_dict(), path + "/deepND_ST_"+diseasename+"_trial"+str(j+1)+"_fold"+str(k1+1)+"_"+str(k2+1)+".pth") 
                # -------------------------------------------------------------
                adjusted_mean_scores = (torch.exp(out.cpu()))[:,1]
                adjusted_mean_scores[data.train_mask] = 0.0
                adjusted_mean_scores[data.auc_mask] = 0.0
                predictions[:,0] += adjusted_mean_scores
                # -------------------------------------------------------------
                area_under_roc = roc_auc_score(data.y.cpu()[data.test_mask],(F.softmax(out.cpu()[data.test_mask, :],dim=1))[:,1])
                aucs.append(area_under_roc)
                print("AUC", area_under_roc)
                
                average_precision = average_precision_score(data.y.cpu()[data.test_mask],(F.softmax(out.cpu()[data.test_mask, :],dim=1))[:,1])   
                aupr.append(average_precision)
                print('Average precision-recall score: {0:0.2f}'.format(average_precision))
                
                u,p = mannwhitneyu((F.softmax(out.cpu()[data.e1mask, :],dim=1))[:,1],(F.softmax(out.cpu()[data.negmask, :],dim=1))[:,1])
                UsandPs.append([u,p])
                
                for i in range(network_count * 4):
                    average_att[i] += torch.mean(model.experts[:,i]).item()
                    stddev_att[i] += model.experts[:,i].std().item()
                    average_att_gold[i] += torch.mean(model.experts[g_bs_tada_intersect_indices,i]).item()
                    stddev_att_gold[i] += model.experts[g_bs_tada_intersect_indices,i].std().item()
                    average_att_gold_e1[i] += torch.mean(model.experts[g_bs_tada_intersect_indices[0:18],i]).item()
                    average_att_gold_e1e2[i] += torch.mean(model.experts[g_bs_tada_intersect_indices[0:49],i]).item()
                    average_att_gold_neg[i] += torch.mean(model.experts[n_bs_tada_intersect_indices,i]).item()
                    
                    att_leak_prevention =  model.experts[:,i]
                    att_leak_prevention[data.train_mask] = 0.0
                    att_leak_prevention[data.validation_mask] = 0.0
                    all_att[i] += att_leak_prevention
                    
                    pre_att_buffer = model.expert_results[i]
                    pre_att_buffer[data.train_mask] = 0.0
                    pre_att_buffer[data.validation_mask] = 0.0
                    pre_att[i] += pre_att_buffer 
                
                # ------------------------------------------------------------- 
        print("-"*10)
        print(diseasename+" Trial Mean AUC:" + str(np.mean(aucs[-20:])))
        print(diseasename+" Trial Mean AUPR:" + str(np.mean(aupr[-20:])))    
        print("-"*10)
        print(diseasename+" Current Median AUC:" + str(np.median(aucs)))
        print(diseasename+" Current Median AUPR:" + str(np.median(aupr)))    
    
        print("-"*80)
    ###############################################################################################################################################    
    # Writing Final Result of the Session
    writePrediction(predictions, g_bs_tada_intersect_indices, n_bs_tada_intersect_indices, root, diseasename= diseasename, trial = trial, k = k)
    riteExperimentSatats( aucs, aupr, root = root, diseasename=diseasename, trial = trial, k = k, init_time =init_time, network_count =network_count , mode = mode)
    
    # HEATMAPS
    heatmap = torch.zeros(4, network_count,dtype=torch.float)
    heatmap2 = torch.zeros(4, network_count,dtype=torch.float)
    heatmap3 = torch.zeros(4, network_count,dtype=torch.float)
    heatmap4 = torch.zeros(4, network_count,dtype=torch.float)
    heatmap5 = torch.zeros(4, network_count,dtype=torch.float)
    heatmap6 = torch.zeros(4, network_count,25825, dtype=torch.float)
    heatmap7 = torch.zeros(25825, network_count * 4, dtype=torch.float)
    heatmap8 = torch.zeros(25825, network_count * 4, dtype=torch.float)
    
    #heatmap[0,:] = average_att[:7]
    for i in range(network_count):
        heatmap[0,i] = average_att[i]
        heatmap2[0,i] = average_att_gold[i]
        heatmap3[0,i] = average_att_gold_e1[i]
        heatmap4[0,i] = average_att_gold_e1e2[i]
        heatmap5[0,i] = average_att_gold_neg[i]
        heatmap6[0,i,:] = all_att[i]
            
    #heatmap[1,2:] = average_att[7:12]
    for i in range(network_count, network_count * 2):
        heatmap[1,i - network_count] = average_att[i]
        heatmap2[1,i - network_count] = average_att_gold[i]
        heatmap3[1,i - network_count] = average_att_gold_e1[i]
        heatmap4[1,i - network_count] = average_att_gold_e1e2[i]
        heatmap5[1,i - network_count] = average_att_gold_neg[i]
        heatmap6[1,i-network_count,:] = all_att[i]
        
    #heatmap[2,:] = average_att[12:19]
    for i in range(network_count * 2, network_count * 3):
        heatmap[2,i - network_count * 2] = average_att[i]
        heatmap2[2,i - network_count * 2] = average_att_gold[i]
        heatmap3[2,i - network_count * 2] = average_att_gold_e1[i]
        heatmap4[2,i - network_count * 2] = average_att_gold_e1e2[i]
        heatmap5[2,i - network_count * 2] = average_att_gold_neg[i]
        heatmap6[2,i-network_count*2,:] = all_att[i]
        
    #heatmap[3,:] = average_att[19:26]
    for i in range(network_count * 3,network_count * 4):
        heatmap[3,i - network_count * 3] = average_att[i]
        heatmap2[3,i - network_count * 3] = average_att_gold[i]
        heatmap3[3,i - network_count * 3] = average_att_gold_e1[i]
        heatmap4[3,i - network_count * 3] = average_att_gold_e1e2[i]
        heatmap5[3,i - network_count * 3] = average_att_gold_neg[i]
        heatmap6[3,i-network_count*3,:] = all_att[i]
        
    torch.save(heatmap, path + "/heatmap_tensor.pt");
    torch.save(heatmap2, path + "/heatmap_gold_tensor.pt");
    torch.save(heatmap3, path + "/heatmap_gold_e1_tensor.pt");
    torch.save(heatmap4, path + "/heatmap_gold_e1e2_tensor.pt");
    torch.save(heatmap5, path + "/heatmap_gold_neg_tensor.pt");
    torch.save(heatmap6, path + "/heatmap_all.pt");
    
    heatmap7 = all_att
    torch.save(heatmap7, path + "/heatmap_flat_all.pt");
    
    heatmap8 = pre_att
    torch.save(heatmap8, path + "/heatmap_pre_att_all.pt");
    
