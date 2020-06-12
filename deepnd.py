"""
deepnd.py
Training and test processes of DeepND
for replicating previous experiments and reproducing data
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

def deepnd(root, path, input_size, mode, l_rate, wd, trial, k, diseasename , devices, pfcgpumask, mdcbcgpumask, shagpumask, v1cgpumask, state, experiment, networks):
    network_count = len(networks)
    geneNames_all = pd.read_csv(root + "Data/row-genes.txt", header = None)
    geneNames_all = geneNames_all[0].tolist()
    geneDict = constructGeneDictionary(root + "Data/hugogenes_entrez.txt")
    gene_names_list = [str(item) for item in geneNames_all]
    
    # GOLD STANDARDS
    # Following section loads gold standard genes
    # To use other standards, following section needs to be changed
    g_bs_tada_intersect_indices_asd, n_bs_tada_intersect_indices_asd, y1, pos_gold_std_evidence_asd, gold_evidence_asd = load_goldstandards(root, geneNames_all, geneDict, diseasename = "ASD")
    g_bs_tada_intersect_indices_id, n_bs_tada_intersect_indices_id, y2, pos_gold_std_evidence_id, gold_evidence_id = load_goldstandards(root, geneNames_all, geneDict, diseasename = "ID")

    asd_e1_gene_indices, asd_e1_perm, asd_e2_gene_indices, asd_e2_perm, asd_e3e4_gene_indices, asd_e3e4_perm, asd_neg_perm, asd_counts = create_validation_set( g_bs_tada_intersect_indices_asd, n_bs_tada_intersect_indices_asd, gold_evidence_asd, k = 5, state = state)
    id_e1_gene_indices, id_e1_perm, id_e2_gene_indices, id_e2_perm, id_e3e4_gene_indices, id_e3e4_perm, id_neg_perm, id_counts = create_validation_set( g_bs_tada_intersect_indices_id, n_bs_tada_intersect_indices_id, gold_evidence_id, k = 5, state = state)

    # FEATURES

    data_asd, featuresasd = loadFeatures(root, y1, geneNames_all, devices, diseasename = "ID")
    data_id, featuresid = loadFeatures(root, y2, geneNames_all, devices, diseasename = "ASD")
    

    commonfeatures = np.load(root + "Data/Multi_TADA_Features.npy")
    commonfeatures = torch.from_numpy(commonfeatures).float()
    commonfeatures = (commonfeatures - torch.mean(commonfeatures,0)) / (torch.std(commonfeatures,0))
    commonfeatures = Data(x=commonfeatures)
    features = commonfeatures.x.to(devices[0]) 
    
    # NETWORKS
    pfcnetworks, pfcnetworkweights, mdcbcnetworks, mdcbcnetworkweights, v1cnetworks, v1cnetworkweights, shanetworks, shanetworkweights = load_networks(root, devices,  pfcgpumask, mdcbcgpumask, shagpumask, v1cgpumask, mask = networks) 
    
    # MODEL CONSTRUCTION
    model = DeepND_ST(devices, pfcgpumask, mdcbcgpumask, shagpumask, v1cgpumask, featsize=input_size, unit=input_size)
    
    #ASD
    average_attasd = []
    average_att_goldasd = []
    stddev_attasd = []
    stddev_att_goldasd  = []
    average_att_gold_e1asd  = []
    average_att_gold_e1e2asd = []
    average_att_gold_negasd = []
    
    for i in range(network_count * 4):
        average_attasd.append(0.0)
        average_att_goldasd.append(0.0)
        stddev_attasd.append(0.0)
        stddev_att_goldasd.append(0.0)
        average_att_gold_e1asd.append(0.0)
        average_att_gold_e1e2asd.append(0.0)
        average_att_gold_negasd.append(0.0)
        
    # ID
    average_attid = []
    average_att_goldid = []
    stddev_attid = []
    stddev_att_goldid  = []
    average_att_gold_e1id  = []
    average_att_gold_e1e2id = []
    average_att_gold_negid = []
    
    for i in range(network_count * 4):
        average_attid.append(0.0)
        average_att_goldid.append(0.0)
        stddev_attid.append(0.0)
        stddev_att_goldid.append(0.0)
        average_att_gold_e1id.append(0.0)
        average_att_gold_e1e2id.append(0.0)
        average_att_gold_negid.append(0.0)
    
    predictions_asd = torch.zeros(commonfeatures.x.shape[0],1)
    predictions_id = torch.zeros(commonfeatures.x.shape[0],1)
    # Emtpy lists for tracking performance metrics and memory usage
    aupr_asd = []
    aucs_asd = [] 
    up_asd = []
    aupr_id = []
    aucs_id = []  
    up_id =[]
    usage = 0
    cached = 0
    
    # Early Stop Configuration
    early_stop_enabled = True
    old_loss = [100,100]
    early_stop_window = 7
    epoch_count = []
    for j in range(trial): # 10 here means Run count. Run given times and calculate average AUC found from each run.
        print("Trial:", j+1)
        
        # Losses
        idtloss=[]
        idvloss=[]
        asdtloss=[]
        asdvloss=[]
        
        fpr = dict()
        tpr = dict()
        
        # Memory Update!
        current_usage = 0
        current_cached = 0
        for d in range(torch.cuda.device_count()):
            current_usage += torch.cuda.max_memory_allocated(device='cuda:'+str(d))
            current_cached += torch.cuda.max_memory_cached(device='cuda:'+str(d))
        usage = max(usage,current_usage)
        cached = max(cached, current_cached)
        print("GPU Memory Usage:", usage / 8**10, "GB Used, ", cached / 8**10, "GB Cached")
        
        for k1 in range(k):
            #ASD
            test_mask_asd = [asd_e1_gene_indices[index] for index in asd_e1_perm[k1 * math.ceil(asd_counts[0]/k): min(asd_counts[0], (k1 + 1) * math.ceil(asd_counts[0]/k)) ] ]
            test_mask_asd +=  [n_bs_tada_intersect_indices_asd[item] for item in asd_neg_perm[k1 * math.ceil(asd_counts[3]/k) : min(asd_counts[3] , (k1 + 1) * math.ceil(asd_counts[3]/k))] ]
            data_asd.test_mask = test_mask_asd.copy()
            
            test_mask_asd += [asd_e2_gene_indices[index] for index in asd_e2_perm[(k1) * math.ceil(asd_counts[1]/k): min(asd_counts[1], (k1 + 1) * math.ceil(asd_counts[1]/k)) ] ] 
            test_mask_asd += [asd_e3e4_gene_indices[index] for index in asd_e3e4_perm[(k1) * math.ceil(asd_counts[2]/k): min(asd_counts[2], (k1 + 1) * math.ceil(asd_counts[2]/k)) ] ] 
            
            asd_k_e1_perm = np.delete(asd_e1_perm,np.s_[k1 * math.ceil(asd_counts[0]/k): min(asd_counts[0] ,(k1 + 1) * math.ceil(asd_counts[0]/k))],axis=0)
            asd_k_neg_perm = np.delete(asd_neg_perm,np.s_[k1 * math.ceil(asd_counts[3]/k): min(asd_counts[3], (k1 + 1) * math.ceil(asd_counts[3]/k)) ],axis=0)
            asd_k_e2_perm = np.delete(asd_e2_perm,np.s_[k1 * math.ceil(asd_counts[1]/k): min(asd_counts[1], (k1 + 1) * math.ceil(asd_counts[1]/k)) ],axis=0)
            asd_k_e3e4_perm = np.delete(asd_e3e4_perm,np.s_[k1 * math.ceil(asd_counts[2]/k): min(asd_counts[2], (k1 + 1) * math.ceil(asd_counts[2]/k)) ],axis=0)
            
            #ID
            test_mask_id = [id_e1_gene_indices[index] for index in id_e1_perm[k1 * math.ceil(id_counts[0]/k): min(id_counts[0], (k1 + 1) * math.ceil(id_counts[0]/k)) ] ]
            test_mask_id +=  [n_bs_tada_intersect_indices_id[item] for item in id_neg_perm[k1 * math.ceil(id_counts[3]/k) : min(id_counts[3] , (k1 + 1) * math.ceil(id_counts[3]/k))] ]
            data_id.test_mask = test_mask_id.copy()
            
            test_mask_id += [id_e2_gene_indices[index] for index in id_e2_perm[(k1) * math.ceil(id_counts[1]/k): min(id_counts[1], (k1 + 1) * math.ceil(id_counts[1]/k)) ] ] 
            test_mask_id += [id_e3e4_gene_indices[index] for index in id_e3e4_perm[(k1) * math.ceil(id_counts[2]/k): min(id_counts[2], (k1 + 1) * math.ceil(id_counts[0]/k)) ] ] 
            
            id_k_e1_perm = np.delete(id_e1_perm,np.s_[k1 * math.ceil(id_counts[0]/k): min(id_counts[0],(k1 + 1) * math.ceil(id_counts[0]/k))],axis=0)
            id_k_neg_perm = np.delete(id_neg_perm,np.s_[k1 * math.ceil(id_counts[3]/k): min(id_counts[3], (k1 + 1) * math.ceil(id_counts[3]/k)) ],axis=0)
            id_k_e2_perm = np.delete(id_e2_perm,np.s_[k1 * math.ceil(id_counts[1]/k): min(id_counts[1], (k1 + 1) * math.ceil(id_counts[1]/k)) ],axis=0)
            id_k_e3e4_perm = np.delete(id_e3e4_perm,np.s_[k1 * math.ceil(id_counts[2]/k): min(id_counts[2], (k1 + 1) * math.ceil(id_counts[2]/k)) ],axis=0)
            
            for k2 in range(k-1): # K-FOLD Cross Validation
                print("Fold", k1+1, "_",  k2+1, "of Trial", j+1)
                
                # Adjust masks - NOTE: Masks contain indices of samples. 
                # Example: if train mask contains 2 genes with indices 6 and 12, train mask should be --> train_mask = [6, 12]
                # Add leftout E1 genes to validation mask
                
                # ASD
                validation_mask_asd = [asd_e1_gene_indices[index] for index in asd_k_e1_perm[(k2) * math.ceil(asd_counts[0]/k): min(asd_counts[0], (k2 + 1) * math.ceil(asd_counts[0]/k)) ] ] 
                print("ASD Validation Mask Length After E1:", len(validation_mask_asd))
                print('ASD Validation Gene(s):', [gene_names_list[i] for i in validation_mask_asd])
                # Add negative genes to validation mask
                validation_mask_asd +=  [n_bs_tada_intersect_indices_asd[item] for item in asd_k_neg_perm[k2 * math.ceil(asd_counts[3]/k) : min(asd_counts[3] , (k2 + 1) * math.ceil(asd_counts[3]/k))] ]
                data_asd.auc_mask = validation_mask_asd.copy()
                #print("Supposed AUC Mask Length:", len(data.auc_mask))
                validation_mask_asd += [asd_e2_gene_indices[index] for index in asd_k_e2_perm[(k2) * math.ceil(asd_counts[1]/k): min(asd_counts[1], (k2 + 1) * math.ceil(asd_counts[1]/k)) ] ] 
                validation_mask_asd += [asd_e3e4_gene_indices[index] for index in asd_k_e3e4_perm[(k2) * math.ceil(asd_counts[2]/k): min(asd_counts[2], (k2 + 1) * math.ceil(asd_counts[2]/k)) ] ] 
                
                # Construct Train Mask
                train_mask_asd = g_bs_tada_intersect_indices_asd + n_bs_tada_intersect_indices_asd
                print("Total Gene Count:", len(train_mask_asd))
                train_mask_asd = [item for item in train_mask_asd if item not in sorted(validation_mask_asd + test_mask_asd)]
                print("ASD Final Validation Mask Length:", len(validation_mask_asd))
                print("ASD Final Train Mask Length:", len(train_mask_asd))
                
                #Uncomment line below if you want to use sample weights
                sample_weights_asd = torch.ones((len(train_mask_asd)), dtype = torch.float).to(devices[0])
                for index, value in enumerate(train_mask_asd):
                    if value in g_bs_tada_intersect_indices_asd:   
                        index2 = g_bs_tada_intersect_indices_asd.index(value)       
                        evidence = pos_gold_std_evidence_asd[index2]
                        if evidence == "E2":
                            sample_weights_asd[index] = 0.5
                        elif evidence == "E3" or evidence == "E4":
                            sample_weights_asd[index] = 0.25
                    elif value in n_bs_tada_intersect_indices_asd:
                        sample_weights_asd[index] = 0.5
                data_asd.train_mask = torch.tensor(train_mask_asd, dtype= torch.long)
                data_asd.validation_mask = torch.tensor(validation_mask_asd, dtype=torch.long)
                    
                # ID
                validation_mask_id = [id_e1_gene_indices[index] for index in id_k_e1_perm[(k2) * math.ceil(id_counts[0]/k): min(id_counts[0], (k2 + 1) * math.ceil(id_counts[0]/k)) ] ] 
                print("ID Validation Mask Length After E1:", len(validation_mask_id))
                print('ID Validation Gene(s):', [gene_names_list[i] for i in validation_mask_id])
                # Add negative genes to validation mask
                validation_mask_id +=  [n_bs_tada_intersect_indices_id[item] for item in id_k_neg_perm[k2 * math.ceil(id_counts[3]/k) : min(id_counts[3] , (k2 + 1) * math.ceil(id_counts[3]/k))] ]
                data_id.auc_mask = validation_mask_id.copy()
                #print("Supposed AUC Mask Length:", len(data.auc_mask))
                validation_mask_id += [id_e2_gene_indices[index] for index in id_k_e2_perm[(k2) * math.ceil(id_counts[1]/k): min(id_counts[1], (k2 + 1) * math.ceil(id_counts[1]/k)) ] ] 
                validation_mask_id += [id_e3e4_gene_indices[index] for index in id_k_e3e4_perm[(k2) * math.ceil(id_counts[2]/k): min(id_counts[2], (k2 + 1) * math.ceil(id_counts[2]/k)) ] ] 
                
                # Construct Train Mask
                train_mask_id = g_bs_tada_intersect_indices_id + n_bs_tada_intersect_indices_id
                print("Total Gene Count:", len(train_mask_id))
                train_mask_id = [item for item in train_mask_id if item not in sorted(validation_mask_id + test_mask_id)]
                print("ID  Final Validation Mask Length:", len(validation_mask_id))
                print("ID Final Train Mask Length:", len(train_mask_id))
                
                #Uncomment line below if you want to use sample weights
                sample_weights_id = torch.ones((len(train_mask_id)), dtype = torch.float).to(devices[0])
                for index, value in enumerate(train_mask_id):
                    if value in g_bs_tada_intersect_indices_id:
                        index2 = g_bs_tada_intersect_indices_id.index(value)
                        evidence = pos_gold_std_evidence_id[index2]
                        if evidence == "E2":
                            sample_weights_id[index] = 0.5
                        elif evidence == "E3" or evidence == "E4":
                            sample_weights_id[index] = 0.25
                    elif value in n_bs_tada_intersect_indices_asd:
                        sample_weights_asd[index] = 0.25
                
                data_id.train_mask = torch.tensor(train_mask_id, dtype= torch.long)
                data_id.validation_mask = torch.tensor(validation_mask_id, dtype=torch.long)
                
                if mode:
                    # Test mode
                    model.load_state_dict(torch.load(root + diseasename + "Exp" + str(experiment) + "/deepND_trial"+str(j+1)+"_fold"+str(k1+1)+"_"+str(k2+1)+".pth"))
                    model = model.eval()
                    with torch.no_grad():
                        out1, out2 = model(features, featuresasd, featuresid, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights, devices, pfcgpumask, mdcbcgpumask, v1cgpumask, shagpumask)
                        _, pred1 = out1.max(dim=1)
                        _, pred2 = out2.max(dim=1)
                        correct1 = pred1[data_asd.auc_mask].eq(data_asd.y1[data_asd.auc_mask]).sum().item()
                        correct2 = pred2[data_id.auc_mask].eq(data_id.y2[data_id.auc_mask]).sum().item()
                        correctTrain1 = pred1[data_asd.train_mask].eq(data_asd.y1[data_asd.train_mask]).sum().item()
                        correctTrain2 = pred2[data_id.train_mask].eq(data_id.y2[data_id.train_mask]).sum().item()
                        acc1 = correct1 / len(data_asd.auc_mask)
                        acc2 = correct2 / len(data_id.auc_mask)
                        accTrain1 = correctTrain1 / len(data_asd.train_mask)
                        accTrain2 = correctTrain2 / len(data_id.train_mask)
                else:
                    # Train mode
                    model.apply(weight_reset)
                    optimizerc = torch.optim.Adam(model.commonmlp.parameters(), lr = l_rate[0], weight_decay = wd )   
                    optimizerasd = torch.optim.Adam( model.ASDBranch.parameters(), lr = l_rate[1], weight_decay = wd )   
                    optimizerid = torch.optim.Adam(model.IDBranch.parameters(), lr = l_rate[2], weight_decay = wd ) 
                    
                    ASDFit = False
                    IDFit = False
                
                    for epoch in range(max_epoch):
                        model = model.train()
                        optimizerc.zero_grad()
                        optimizerasd.zero_grad()
                        optimizerid.zero_grad()
                        out1,out2 = model(features, featuresasd, featuresid, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights, devices, pfcgpumask, mdcbcgpumask, v1cgpumask, shagpumask)
                        
                        # You can adjust class weights using values in FloatTensor
                        loss1 = F.nll_loss(out1[data_asd.train_mask], data_asd.y1[data_asd.train_mask], weight = torch.FloatTensor([1.0, 1.0]).to(devices[0]), reduction ='none')
                        loss2 = F.nll_loss(out2[data_id.train_mask], data_id.y2[data_id.train_mask], weight = torch.FloatTensor([1.0, 1.0]).to(devices[0]), reduction ='none')
                        
                        loss1 = (loss1 * sample_weights_asd).mean()
                        loss2 = (loss2 * sample_weights_id).mean()
                        # As we have multiple branches, we set the "retain_graph=True"
                        loss1.backward(retain_graph=True)
                        loss2.backward(retain_graph=True)
                        
                        asdtloss.append(loss1.cpu().item())
                        idtloss.append(loss2.cpu().item())
                        
                        optimizerc.step()
                        optimizerasd.step()
                        optimizerid.step()
                        #----------------------------------------------------------------------------------------#
                        model = model.eval()
                        with torch.no_grad():
                            out1, out2 = model(features, featuresasd, featuresid, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights, devices, pfcgpumask, mdcbcgpumask, v1cgpumask, shagpumask)
                            _, pred1 = out1.max(dim=1)
                            _, pred2 = out2.max(dim=1)
                            correct1 = pred1[data_asd.auc_mask].eq(data_asd.y1[data_asd.auc_mask]).sum().item()
                            correct2 = pred2[data_id.auc_mask].eq(data_id.y2[data_id.auc_mask]).sum().item()
                            correctTrain1 = pred1[data_asd.train_mask].eq(data_asd.y1[data_asd.train_mask]).sum().item()
                            correctTrain2 = pred2[data_id.train_mask].eq(data_id.y2[data_id.train_mask]).sum().item()
                            acc1 = correct1 / len(data_asd.auc_mask)
                            acc2 = correct2 / len(data_id.auc_mask)
                            accTrain1 = correctTrain1 / len(data_asd.train_mask)
                            accTrain2 = correctTrain2 / len(data_id.train_mask)
                            
                            valLoss = [F.nll_loss(out1[data_asd.auc_mask], data_asd.y1[data_asd.auc_mask]).mean(), F.nll_loss(out2[data_id.auc_mask], data_id.y2[data_id.auc_mask]).mean()]
                            asdvloss.append(valLoss[0].cpu().item())
                            idvloss.append(valLoss[1].cpu().item())
                        
                        if epoch != 0 and epoch % 25 == 0:
                            print('Val Acc:{:.4f}| Val Loss: ASD {:.4f}, ID {:.4f}| Train Acc: {:.4f}'.format(np.mean([acc1,acc2]),valLoss[0],valLoss[1], np.mean([accTrain1,accTrain2])))
        
                        # Early Stop Checks                
                        if early_stop_enabled:
                            if valLoss[0] < old_loss[0]: # ASD Early fit
                                early_stop_count_asd = 0
                                old_loss[0] = valLoss[0]
                            else:
                                early_stop_count_asd += 1
                                
                            if early_stop_count_asd >= early_stop_window:
                                if ASDFit:
                                    model.ASDBranch.apply(freeze_layer)
                                    model.commonmlp.apply(freeze_layer)
                                    optimizerasd = torch.optim.Adam( model.ASDBranch.parameters(), lr=0.0, weight_decay=wd ) 
                                    optimizerc = torch.optim.Adam(model.commonmlp.parameters(), lr=0.0, weight_decay=wd )
                                    optimizerc = torch.optim.Adam(model.commonmlp.parameters(), lr=0.0, weight_decay=wd )
                                    print('ASD freezed! Val Loss: ASD {:.4f}, ID {:.4f}'.format(valLoss[0],valLoss[1]))
                                    early_stop_count_asd = float("-inf")                            
                                else:
                                    early_stop_count_asd = 0
                                    print("Epoch:", epoch,", Loss ASD:",loss1.mean().item(),", Loss ID:",loss2.mean().item())
                                    print("ASD slowed down!")                            
                                    optimizerasd = torch.optim.Adam( model.ASDBranch.parameters(), lr=lrasd/20.0, weight_decay=wd ) 
                                    optimizerc = torch.optim.Adam(model.commonmlp.parameters(), lr=lrc/20.0, weight_decay=wd )
                                    ASDFit = True
                                
                            if valLoss[1] < old_loss[1]: # ID Early fit
                                early_stop_count_id = 0
                                old_loss[1] = valLoss[1]
                            else:
                                early_stop_count_id += 1 
                                
                            if early_stop_count_id >= early_stop_window:
                                if IDFit:
                                    model.IDBranch.apply(freeze_layer)
                                    model.commonmlp.apply(freeze_layer)
                                    optimizerid = torch.optim.Adam( model.IDBranch.parameters(), lr=0.0, weight_decay=wd ) 
                                    optimizerc = torch.optim.Adam(model.commonmlp.parameters(), lr=0.0, weight_decay=wd )
                                    print('ID freezed! Val Loss: ASD {:.4f}, ID {:.4f}'.format(valLoss[0],valLoss[1]))
                                    early_stop_count_id = float("-inf")   
                                else:
                                    early_stop_count_id = 0
                                    print("Epoch:", epoch,", Loss ASD:",loss1.mean().item(),", Loss ID:",loss2.mean().item())
                                    print("ID slowed down!")    
                                    optimizerid = torch.optim.Adam(model.IDBranch.parameters(), lr = lrid/20.0, weight_decay=wd ) 
                                    optimizerc = torch.optim.Adam(model.commonmlp.parameters(), lr = lrc/20.0, weight_decay=wd )
                                    IDFit = True
                                
                            if IDFit and ASDFit: 
                                print("Epoch:", epoch,", Loss ASD:",loss1.mean().item(),", Loss ID:",loss2.mean().item())
                                print("Training Done!")
                                # Saving the model with the recommended method on "https://pytorch.org/tutorials/beginner/saving_loading_models.html"
                                torch.save(model.state_dict(), path + "/deepND_trial"+str(j+1)+"_fold"+str(k1+1)+"_"+str(k2+1)+".pth")    
                                epoch_count.append(epoch)
                                break
                    
                # -------------------------------------------------------------
                adjusted_mean_scores = (F.softmax(out1.cpu(),dim=1))[:,1]
                adjusted_mean_scores[data_asd.train_mask] = 0.0
                adjusted_mean_scores[data_asd.validation_mask] = 0.0
                predictions_asd[:,0] += adjusted_mean_scores
        
                adjusted_mean_scores = (F.softmax(out2.cpu(),dim=1))[:,1]
                adjusted_mean_scores[data_id.train_mask] = 0.0
                adjusted_mean_scores[data_id.validation_mask] = 0.0
                predictions_id[:,0] += adjusted_mean_scores
                # -------------------------------------------------------------
                area_under_roc = roc_auc_score(data_asd.y1.cpu()[data_asd.test_mask],(F.softmax(out1.cpu()[data_asd.test_mask, :],dim=1))[:,1])
                aucs_asd.append(area_under_roc)                                            
                print("ASD AUC", aucs_asd[-1])
                aupr_asd.append(average_precision_score(data_asd.y1.cpu()[data_asd.test_mask],(F.softmax(out1.cpu()[data_asd.test_mask, :],dim=1))[:,1]))
                print("ASD AUPR", aupr_asd[-1])
                
                u,p = mannwhitneyu((F.softmax(out1.cpu()[data_asd.e1mask, :],dim=1))[:,1],(F.softmax(out1.cpu()[data_asd.negmask, :],dim=1))[:,1])
                up_asd.append([u,p])
                
                for i in range(network_count * 4):
                    average_attasd[i] += torch.mean(model.ASDBranch.experts[:,i]).item()
                    stddev_attasd[i] += model.ASDBranch.experts[:,i].std().item()
                    average_att_goldasd[i] += torch.mean(model.ASDBranch.experts[g_bs_tada_intersect_indices_asd,i]).item()
                    stddev_att_goldasd[i] += model.ASDBranch.experts[g_bs_tada_intersect_indices_asd,i].std().item()
                    average_att_gold_e1asd[i] += torch.mean(model.ASDBranch.experts[g_bs_tada_intersect_indices_asd[0:46],i]).item()
                    average_att_gold_e1e2asd[i] += torch.mean(model.ASDBranch.experts[g_bs_tada_intersect_indices_asd[0:46+67],i]).item()
                    average_att_gold_negasd[i] += torch.mean(model.ASDBranch.experts[n_bs_tada_intersect_indices_asd,i]).item()
                    
                    att_leak_prevention =  model.ASDBranch.experts[:,i]
                    att_leak_prevention[data_asd.train_mask] = 0.0
                    att_leak_prevention[data_asd.validation_mask] = 0.0
                    all_att_asd[i] += att_leak_prevention
                    
                    pre_att_buffer_asd = model.ASDBranch.expert_results[i]
                    pre_att_buffer_asd[data_asd.train_mask] = 0.0
                    pre_att_buffer_asd[data_asd.validation_mask] = 0.0
                    pre_att_asd[i] += pre_att_buffer_asd 
                
                # -------------------------------------------------------------
                area_under_roc = roc_auc_score(data_id.y2.cpu()[data_id.test_mask],(F.softmax(out2.cpu()[data_id.test_mask, :],dim=1))[:,1])
                aucs_id.append(area_under_roc)                                            
                print("ID AUC", area_under_roc)
                aupr_id.append(average_precision_score(data_id.y2.cpu()[data_id.test_mask],(F.softmax(out2.cpu()[data_id.test_mask, :],dim=1))[:,1]))
                print("ID AUPR", aupr_id[-1])
                
                u,p = mannwhitneyu((F.softmax(out2.cpu()[data_id.e1mask, :],dim=1))[:,1],(F.softmax(out2.cpu()[data_id.negmask, :],dim=1))[:,1])
                up_id.append([u,p])
    
                for i in range(network_count * 4):
                    average_attid[i] += torch.mean(model.IDBranch.experts[:,i]).item()
                    stddev_attid[i] += model.IDBranch.experts[:,i].std().item()
                    average_att_goldid[i] += torch.mean(model.IDBranch.experts[g_bs_tada_intersect_indices_id,i]).item()
                    stddev_att_goldid[i] += model.IDBranch.experts[g_bs_tada_intersect_indices_id,i].std().item()
                    average_att_gold_e1id[i] += torch.mean(model.IDBranch.experts[g_bs_tada_intersect_indices_id[0:56],i]).item()
                    average_att_gold_e1e2id[i] += torch.mean(model.IDBranch.experts[g_bs_tada_intersect_indices_id[0:56+181],i]).item()
                    average_att_gold_negid[i] += torch.mean(model.IDBranch.experts[n_bs_tada_intersect_indices_id,i]).item()
                    
                    att_leak_prevention =  model.IDBranch.experts[:,i]
                    att_leak_prevention[data_id.train_mask] = 0.0
                    att_leak_prevention[data_id.validation_mask] = 0.0
                    all_att_id[i] += att_leak_prevention
                    
                    pre_att_buffer_id = model.IDBranch.expert_results[i]
                    pre_att_buffer_id[data_id.train_mask] = 0.0
                    pre_att_buffer_id[data_id.validation_mask] = 0.0
                    pre_att_id[i] += pre_att_buffer_id
                
                # -------------------------------------------------------------
                print("."*10)
                print("ASD Current Median AUC:" + str(np.median(aucs_asd)))
                print("ASD Current Median AUPR:" + str(np.median(aupr_asd)))    
                print("ID Current Median AUC:" + str(np.median(aucs_id)))
                print("ID Current Median AUPR:" + str(np.median(aupr_id))) 
                print("-"*10)
            
        # -------------------------------------------------------------
        print("ASD Trial Median AUC :" + str( np.median( aucs_asd[- (k*(k-1)) :] )))
        print("ASD Trial Median AUPR:" + str( np.median( aupr_asd[- (k*(k-1)) :] )))    
        print("ID Trial Median AUC :" + str( np.median( aucs_id[- (k*(k-1)) :] )))
        print("ID Trial Median AUPR:" + str( np.median( aupr_id[- (k*(k-1)) :] )))
        
        print("-"*80)
    
    ###############################################################################################################################################    
    """Writing Final Result of the Session"""
    ###############################################################################################################################################
    
    #ASD final Predictions
    writePrediction(predictions_asd, g_bs_tada_intersect_indices_asd, n_bs_tada_intersect_indices_asd,  path = path, diseasename="ASD", trial = trial, k = k
    
    #ID final Predictions
    writePrediction(predictions_id, g_bs_tada_intersect_indices_id, n_bs_tada_intersect_indices_id,  path = path, diseasename="ID", trial = trial, k = k)
    
    #Experiment Stats
    writeExperimentStats( [aucs_asd, aucs_id], [aupr_asd, aupr_id], path = path , diseasename="Multi", trial = trial, k = k, init_time = init_time, network_count = network_count, mode = mode)
    
    ###############################################################################################################################################
    """HEATMAPS"""
    ###############################################################################################################################################
    ## ASD
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
        heatmap[0,i] = average_attasd[i]
        heatmap2[0,i] = average_att_goldasd[i]
        heatmap3[0,i] = average_att_gold_e1asd[i]
        heatmap4[0,i] = average_att_gold_e1e2asd[i]
        heatmap5[0,i] = average_att_gold_negasd[i]
        heatmap6[0,i,:] = all_att_asd[i]
        
        
    #heatmap[1,2:] = average_att[7:12]
    for i in range(network_count, network_count * 2):
        heatmap[1,i - network_count] = average_attasd[i]
        heatmap2[1,i - network_count] = average_att_goldasd[i]
        heatmap3[1,i - network_count] = average_att_gold_e1asd[i]
        heatmap4[1,i - network_count] = average_att_gold_e1e2asd[i]
        heatmap5[1,i - network_count] = average_att_gold_negasd[i]
        heatmap6[1,i-network_count,:] = all_att_asd[i]
        
    #heatmap[2,:] = average_att[12:19]
    for i in range(network_count,network_count * 3):
        heatmap[2,i - network_count * 2] = average_attasd[i]
        heatmap2[2,i - network_count * 2] = average_att_goldasd[i]
        heatmap3[2,i - network_count * 2] = average_att_gold_e1asd[i]
        heatmap4[2,i - network_count * 2] = average_att_gold_e1e2asd[i]
        heatmap5[2,i - network_count * 2] = average_att_gold_negasd[i]
        heatmap6[2,i-network_count*2,:] = all_att_asd[i]
    #heatmap[3,:] = average_att[19:26]
    for i in range(network_count * 3,network_count * 4):
        heatmap[3,i - network_count * 3] = average_attasd[i]
        heatmap2[3,i - network_count * 3] = average_att_goldasd[i]
        heatmap3[3,i - network_count * 3] = average_att_gold_e1asd[i]
        heatmap4[3,i - network_count * 3] = average_att_gold_e1e2asd[i]
        heatmap5[3,i - network_count * 3] = average_att_gold_negasd[i]
        heatmap6[3,i-network_count*3,:] = all_att_asd[i]
        
    torch.save(heatmap, path + "/heatmap_tensor_asd.pt");
    torch.save(heatmap2, path + "/heatmap_gold_tensor_asd.pt");
    torch.save(heatmap3, path + "/heatmap_gold_e1_tensor_asd.pt");
    torch.save(heatmap4, path + "/heatmap_gold_e1e2_tensor_asd.pt");
    torch.save(heatmap5, path + "/heatmap_gold_neg_tensor_asd.pt");
    torch.save(heatmap6, path + "/heatmap_all_asd.pt");
    
    heatmap7 = all_att_asd
    torch.save(heatmap7, path + "/heatmap_flat_all_asd.pt");
    
    heatmap8 = pre_att_asd
    torch.save(heatmap8, path + "/heatmap_pre_att_asd.pt");
    ## ID
    heatmap = torch.zeros(4, network_count,dtype=torch.float)
    heatmap2 = torch.zeros(4, network_count,dtype=torch.float)
    heatmap3 = torch.zeros(4, network_count,dtype=torch.float)
    heatmap4 = torch.zeros(4, network_count,dtype=torch.float)
    heatmap5 = torch.zeros(4, network_count,dtype=torch.float)
    heatmap6 = torch.zeros(4, network_count, 25825,dtype=torch.float)
    heatmap7 = torch.zeros(25825, network_count * 4, dtype=torch.float)
    heatmap8 = torch.zeros(25825, network_count * 4, dtype=torch.float)
    
    #heatmap[0,:] = average_att[:7]
    for i in range(network_count):
        heatmap[0,i] = average_attid[i]
        heatmap2[0,i] = average_att_goldid[i]
        heatmap3[0,i] = average_att_gold_e1id[i]
        heatmap4[0,i] = average_att_gold_e1e2id[i]
        heatmap5[0,i] = average_att_gold_negid[i]
        heatmap6[0,i,:] = all_att_id[i]
        
    #heatmap[1,2:] = average_att[7:12]
    for i in range(network_count, network_count * 2):
        heatmap[1,i - network_count] = average_attid[i]
        heatmap2[1,i - network_count] = average_att_goldid[i]
        heatmap3[1,i - network_count] = average_att_gold_e1id[i]
        heatmap4[1,i - network_count] = average_att_gold_e1e2id[i]
        heatmap5[1,i - network_count] = average_att_gold_negid[i]
        heatmap6[1,i-network_count,:] = all_att_id[i]
        
    #heatmap[2,:] = average_att[12:19]
    for i in range(network_count,network_count * 3):
        heatmap[2,i - network_count * 2] = average_attid[i]
        heatmap2[2,i - network_count * 2] = average_att_goldid[i]
        heatmap3[2,i - network_count * 2] = average_att_gold_e1id[i]
        heatmap4[2,i - network_count * 2] = average_att_gold_e1e2id[i]
        heatmap5[2,i - network_count * 2] = average_att_gold_negid[i]
        heatmap6[2,i-network_count*2,:] = all_att_id[i]
        
    #heatmap[3,:] = average_att[19:26]
    for i in range(network_count * 3,network_count * 4):
        heatmap[3,i - network_count * 3] = average_attid[i]
        heatmap2[3,i - network_count * 3] = average_att_goldid[i]
        heatmap3[3,i - network_count * 3] = average_att_gold_e1id[i]
        heatmap4[3,i - network_count * 3] = average_att_gold_e1e2id[i]
        heatmap5[3,i - network_count * 3] = average_att_gold_negid[i]
        heatmap6[3,i-network_count*3,:] = all_att_id[i]
        
    torch.save(heatmap, path + "/heatmap_tensor_id.pt");
    torch.save(heatmap2, path + "/heatmap_gold_tensor_id.pt");
    torch.save(heatmap3, path + "/heatmap_gold_e1_tensor_id.pt");
    torch.save(heatmap4, path + "/heatmap_gold_e1e2_tensor_id.pt");
    torch.save(heatmap5, path + "/heatmap_gold_neg_tensor_id.pt");
    torch.save(heatmap6, path + "/heatmap_all_id.pt");
    
    heatmap7 = all_att_id
    torch.save(heatmap7, path + "/heatmap_flat_all_id.pt");
    
    heatmap8 = pre_att_id
    torch.save(heatmap8, path + "/heatmap_pre_att_id.pt");
    