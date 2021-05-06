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
from torch_geometric.data import Data
from torch.autograd import Variable
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import matthews_corrcoef
import time
from models  import *
from utils import *

class DeepND_Driver:
    def __init__(self, root, input_size, mode, l_rate, wd, hidden_units, trial, k, disordername, devices, gpumask, state, experiment, networks, positive_ground_truths, negative_ground_truths, features, verbose, system_gpu_mask, network_gpu_mask, common_layer_units, task_names, feature_names, moe_features=None, moe_feat_size=None):
        self.init_time = time.time()
        self.root = root
        self.input_size = input_size
        self.mode = mode
        self.l_rate = l_rate
        self.trial = trial
        self.k = k
        self.wd = wd
        self.hidden_units = hidden_units
        self.disordername = disordername
        self.devices = devices
        self.network_gpu_mask = gpumask
        self.state = state
        self.experiment = experiment
        self.networks = networks
        self.positive_ground_truths = positive_ground_truths
        self.negative_ground_truths = negative_ground_truths
        self.features = features
        self.verbose = verbose
        self.system_gpu_mask = system_gpu_mask
        self.network_gpu_mask = network_gpu_mask
        self.common_layer_units = common_layer_units        
        self.instance_count = features[0].shape[0]
        self.task_names = task_names
        self.feature_names = feature_names
        if moe_features is not None:
            self.moe_features = moe_features
            self.moe_input_size = moe_feat_size
        else:
            self.moe_features = features
            self.moe_input_size = input_size
        
        
        #Determine number of tasks
        if len(self.input_size) == 1:
            self.task_count = 1
        else:
            self.task_count = len(self.input_size) - 1
        task_names_list = [] 
        
        #Process task names
        if task_names == "indices":
            for i in range(task_count):
                task_names_list.append("Task ", i)
        else:
            task_name_tokens = task_names.split(",")
            for token in task_name_tokens:
                task_names_list.append(token.strip())
            
        if len(task_names_list) != self.task_count:
            print("Provided task names length do not match with the number of tasks. Execution is aborted.")
            exit(0)
                
        #TODO : check all task_count related stuff here to make sure the length of all parameters are OK.
        
        self.network_count = len(networks)
        
        if verbose:
            print("Reading and processing ground truth files")
        gene_indices, gene_permutations, gene_counts, labels = create_gt_list(self.root, self.positive_ground_truths, self.negative_ground_truths, self.verbose, self.k, self.state, self.instance_count)
        if verbose:
            print("Sending features to all GPUs")
        # For each feature set, we are sending it to all GPUs and keep them as a dictionary
        
        gpu_features = []
        for i in range(len(features)):
            feature_dict = {}
            for device in devices:
                feature_dict[device] = features[i].to(device)
                #features[i] = features[i].to(device)
            gpu_features.append(feature_dict)  
            
        gpu_moe_features = []
        for i in range(len(moe_features)):
            feature_dict = {}
            for device in devices:
                feature_dict[device] = moe_features[i].to(device)
            gpu_moe_features.append(feature_dict)  
          
        if self.network_gpu_mask == "auto":
            if self.verbose:
                print("Automatic network assignment option is selected. Networks are being assigned to GPUs automatically.")
            gpu_mask,lowest_load_gpu = assign_networks(self.networks, self.system_gpu_mask, self.verbose)
        else:
            gpu_mask = []
            lowest_load_gpu = 0
            assignment = self.system_gpu_mask.split(",")
            for gpu in assignment:
                gpu_mask.append(int(gpu.strip()))
                
                
        if verbose:
            print("Sending networks to assigned GPUs.")
            
        for i in range(self.network_count):
            networks[i] = networks[i].to(devices[gpu_mask[i]])

        
        if self.verbose:
            print("Sending labels to GPU ", lowest_load_gpu)
        for i in range(len(labels)):
            labels[i] = labels[i].to(devices[lowest_load_gpu])
            
        
        if verbose:
            print("Initializing the model")
        
        model = DeepND(self.devices, gpu_mask, self.input_size, self.common_layer_units, self.hidden_units, self.networks, lowest_load_gpu, self.root, self.experiment, self.disordername, self.moe_input_size )
        
        # Values that will be updated throughout the training process
        average_expert_weights = []
        average_expert_probabilities = []
        aucs = []
        auprcs = []
        mccs = []
        predictions = []
        for task_index in range(self.task_count):
            predictions.append(torch.zeros(self.instance_count,dtype=torch.float))
            
        
        
        #Memory usage information
        usage = 0
        cached = 0
        
        for i in range(self.task_count):
            task_expert_weights = torch.zeros((self.instance_count, self.network_count), dtype = torch.float)
            task_expert_probabilities = torch.zeros((self.instance_count, self.network_count), dtype = torch.float)
            task_aucs = []
            task_auprcs = []
            task_mccs = []
                       
            average_expert_weights.append(task_expert_weights)
            average_expert_probabilities.append(task_expert_probabilities)
            aucs.append(task_aucs)
            auprcs.append(task_auprcs)
            mccs.append(task_mccs)
            
        # Early Stop Configuration
        early_stop_enabled = True #TODO: parameterize this as well
        old_loss = []
        early_stop_window = 7
        for i in range(self.task_count):
            old_loss.append(100.0)
            #epoch_count.append(0)
            predictions.append(torch.zeros(self.instance_count,1))
        

        for j in range(self.trial):
            print("Trial:", j+1)
            # Losses
            training_losses = []
            validation_losses = []
            for task_index in range(self.task_count):
                training_losses.append([])
                validation_losses.append([])
            
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
            
            
            geneNames_all = pd.read_csv(root + "Data/row-genes.txt", header = None)
            geneNames_all = geneNames_all[0].tolist()
            gene_names_list = [str(item) for item in geneNames_all]
            
            for k1 in range(k):
                test_masks = []
                left_out_folds = []
                branch_k_e1_permutations = []
                branch_k_e2_permutations = []
                branch_k_e3e4_permutations = []
                branch_k_neg_permutations = []
                for task_index in range(self.task_count):
                    left_out_test_fold = [gene_indices[task_index][0][index] for index in gene_permutations[task_index][0][k1 * math.ceil(gene_counts[task_index][0]/k): min(gene_counts[task_index][0], (k1 + 1) * math.ceil(gene_counts[task_index][0]/k)) ] ]
                    left_out_test_fold +=  [gene_indices[task_index][-1][item] for item in gene_permutations[task_index][-1][k1 * math.ceil(gene_counts[task_index][-1]/k) : min(gene_counts[task_index][-1] , (k1 + 1) * math.ceil(gene_counts[task_index][-1]/k))] ]
                    test_mask_branch = left_out_test_fold.copy()
                    left_out_test_fold += [gene_indices[task_index][1][index] for index in gene_permutations[task_index][1][(k1) * math.ceil(gene_counts[task_index][1]/k): min(gene_counts[task_index][1], (k1 + 1) * math.ceil(gene_counts[task_index][1]/k)) ] ] 
                    left_out_test_fold += [gene_indices[task_index][2][index] for index in gene_permutations[task_index][2][(k1) * math.ceil(gene_counts[task_index][2]/k): min(gene_counts[task_index][2], (k1 + 1) * math.ceil(gene_counts[task_index][2]/k)) ] ] 
                    test_mask_branch = torch.tensor(test_mask_branch, dtype= torch.long)
                    branch_k_e1_perm = np.delete(gene_permutations[task_index][0],np.s_[k1 * math.ceil(gene_counts[task_index][0]/k): min(gene_counts[task_index][0] ,(k1 + 1) * math.ceil(gene_counts[task_index][0]/k))],axis=0)
                    branch_k_neg_perm = np.delete(gene_permutations[task_index][-1],np.s_[k1 * math.ceil(gene_counts[task_index][3]/k): min(gene_counts[task_index][3], (k1 + 1) * math.ceil(gene_counts[task_index][3]/k)) ],axis=0)
                    branch_k_e2_perm = np.delete(gene_permutations[task_index][1],np.s_[k1 * math.ceil(gene_counts[task_index][1]/k): min(gene_counts[task_index][1], (k1 + 1) * math.ceil(gene_counts[task_index][1]/k)) ],axis=0)
                    branch_k_e3e4_perm = np.delete(gene_permutations[task_index][2],np.s_[k1 * math.ceil(gene_counts[task_index][2]/k): min(gene_counts[task_index][2], (k1 + 1) * math.ceil(gene_counts[task_index][2]/k)) ],axis=0)
                    test_masks.append(test_mask_branch.clone().detach())
                    left_out_folds.append(left_out_test_fold.copy())
                    branch_k_e1_permutations.append(np.copy(branch_k_e1_perm))
                    branch_k_e2_permutations.append(np.copy(branch_k_e2_perm))
                    branch_k_e3e4_permutations.append(np.copy(branch_k_e3e4_perm))
                    branch_k_neg_permutations.append(np.copy(branch_k_neg_perm))
                    print(task_names_list[task_index], " test mask length:", len(left_out_test_fold))
                    
                for k2 in range(k-1): # K-FOLD Cross Validation
                    print("Fold", k1+1, "_",  k2+1, "of Trial", j+1)
                    validation_masks = []
                    left_out_validation_masks = []
                    training_masks = []
                    sample_weights = []
                    for task_index in range(self.task_count):
                        left_out_validation_fold = [gene_indices[task_index][0][index] for index in branch_k_e1_permutations[task_index][(k2) * math.ceil(gene_counts[task_index][0]/k): min(gene_counts[task_index][0], (k2 + 1) * math.ceil(gene_counts[task_index][0]/k)) ] ] 
                        print(task_names_list[task_index], " validation mask length after adding E1:", len(left_out_validation_fold))
                        print(task_names_list[task_index], ' validation gene(s):', [gene_names_list[i] for i in left_out_validation_fold])
                        # Add negative genes to validation mask
                        left_out_validation_fold +=  [gene_indices[task_index][-1][item] for item in branch_k_neg_permutations[task_index][k2 * math.ceil(gene_counts[task_index][3]/k) : min(gene_counts[task_index][3] , (k2 + 1) * math.ceil(gene_counts[task_index][3]/k))] ]
                        validation_mask = left_out_validation_fold.copy()
                        left_out_validation_fold += [gene_indices[task_index][1][index] for index in branch_k_e2_permutations[task_index][(k2) * math.ceil(gene_counts[task_index][1]/k): min(gene_counts[task_index][1], (k2 + 1) * math.ceil(gene_counts[task_index][1]/k)) ] ] 
                        left_out_validation_fold += [gene_indices[task_index][2][index] for index in branch_k_e3e4_permutations[task_index][(k2) * math.ceil(gene_counts[task_index][2]/k): min(gene_counts[task_index][2], (k2 + 1) * math.ceil(gene_counts[task_index][2]/k)) ] ] 
                        
                        # Construct Train Mask

                        train_mask = gene_indices[task_index][0] + gene_indices[task_index][1] + gene_indices[task_index][2] + gene_indices[task_index][-1]
                        print("Total gold standard genes:", len(train_mask))
                        train_mask = [item for item in train_mask if item not in sorted(left_out_validation_fold + left_out_folds[task_index])]
                        print(task_names_list[task_index], " final validation mask length:", len(left_out_validation_fold))
                        print(task_names_list[task_index], " final train mask length:", len(train_mask))
                        
                        sample_weight = torch.ones((len(train_mask)), dtype = torch.float).to(devices[lowest_load_gpu])
                        for index, value in enumerate(train_mask):
                            if value in gene_indices[task_index][1]:
                                sample_weight[index] = 0.5
                            elif value in gene_indices[task_index][2]:
                                sample_weight[index] = 0.25
                            elif value in gene_indices[task_index][-1]:
                                sample_weight[index] = 1
                                
                            #if task_index == 1 and value in gene_indices[task_index][0]:
                            #    sample_weight[index] *= 3
                        train_mask = torch.tensor(train_mask, dtype= torch.long)
                        validation_mask = torch.tensor(validation_mask, dtype=torch.long)
                        validation_masks.append(validation_mask.clone().detach())
                        left_out_validation_masks.append(left_out_validation_fold.copy())
                        training_masks.append(train_mask.clone().detach())
                        sample_weights.append(sample_weight.clone().detach())
                        
                        
                    if mode: # Test mode
                        # Test mode
                        model.load_state_dict(torch.load(root + disordername + "Exp" + str(experiment) + "/deepND_trial"+str(j+1)+"_fold"+str(k1+1)+"_"+str(k2+1)+".pth"))
                        model = model.eval()
                        with torch.no_grad():
                            output = model(gpu_features, gpu_moe_features,  networks, devices, gpu_mask, lowest_load_gpu)
                            preds = []
                            corrects = []
                            corrects_train = []
                            accuracies = []
                            accuracies_train = []
                            for task_index in range(self.task_count):
                                _,pred = output[task_index][0].max(dim=1)
                                preds.append(pred)
                                corrects.append(preds[task_index][validation_masks[task_index]].eq(labels[task_index][validation_masks[task_index]]).sum().item())
                                corrects_train.append(preds[task_index][training_masks[task_index]].eq(labels[task_index][training_masks[task_index]]).sum().item())
                                accuracies.append(corrects[task_index] / len(validation_masks[task_index]))
                                accuracies_train.append(corrects_train[task_index] / len(training_masks[task_index]))
                    else: # Training mode
                        model.apply(weight_reset)
                        
                        optimizers = []
                        branch_fit = []
                        early_stop_counts = []
                        if self.task_count == 1:
                            optimizers.append(torch.optim.Adam(model.branches[0].parameters(), lr = l_rate[0], weight_decay = wd))
                            branch_fit.append(False)
                        else:
                            
                            for task_index in range(self.task_count):
                                optimizers.append(torch.optim.Adam(model.branches[task_index].parameters(), lr = l_rate[task_index], weight_decay = wd))
                                branch_fit.append(False)
                            optimizers.append(torch.optim.Adam(model.commonmlp.parameters(),lr = l_rate[-1], weight_decay = wd))
                        old_loss = []                        
                        for task_index in range(self.task_count):
                            early_stop_counts.append(0)
                            old_loss.append(100.0)
                            
                            
                        for epoch in range(1000):
                            model = model.train()
                            for task_index in range(self.task_count):
                                optimizers[task_index].zero_grad()
                            if self.task_count > 1:
                                optimizers[-1].zero_grad()
                                

                            output = model(gpu_features, gpu_moe_features, networks, devices, gpu_mask, lowest_load_gpu)
                            losses = []
                            
                            for task_index in range(self.task_count):
                                losses.append(F.nll_loss(output[task_index][0][training_masks[task_index]], labels[task_index][training_masks[task_index]], reduction = 'none'))
                                losses[task_index] = (losses[task_index] * sample_weights[task_index]).mean()
                                # As we have multiple branches, we set the "retain_graph=True"
                                losses[task_index].backward(retain_graph = True)
                                training_losses[task_index].append(losses[task_index].cpu().item())
                                optimizers[task_index].step()
                            if self.task_count > 1:
                                optimizers[-1].step()
                            
                            #----------------------------------------------------------------------------------------#
                            model = model.eval()
                            with torch.no_grad():
                                output = model(gpu_features, gpu_moe_features, networks, devices, gpu_mask, lowest_load_gpu)
                                preds = []
                                corrects = []
                                corrects_train = []
                                accuracies = []
                                accuracies_train = []
                                for task_index in range(self.task_count):
                                    _,pred = output[task_index][0].max(dim=1)
                                    preds.append(pred)
                                    corrects.append(preds[task_index][validation_masks[task_index]].eq(labels[task_index][validation_masks[task_index]]).sum().item())
                                    corrects_train.append(preds[task_index][training_masks[task_index]].eq(labels[task_index][training_masks[task_index]]).sum().item())
                                    accuracies.append(corrects[task_index] / len(validation_masks[task_index]))
                                    accuracies_train.append(corrects_train[task_index] / len(training_masks[task_index]))
                                    
                                
                                val_loss = [F.nll_loss(output[task_index][0][validation_masks[task_index]], labels[task_index][validation_masks[task_index]]).mean() for task_index in range(self.task_count)]
                                for task_index in range(self.task_count):
                                    validation_losses[task_index].append(val_loss[task_index].cpu().item())
                                
                            if epoch != 0 and epoch % 25 == 0:
                                epoch_text = ""
                                for task_index in range(self.task_count):
                                    #print(task_names_list[task_index], " Val Acc:{:.4f}| Val Loss: {:.4f} | Train Acc: {:.4f} | Train Loss: {:.4f} ".format(accuracies[task_index], val_loss[task_index], accuracies_train[task_index], losses[task_index]))
                                    epoch_text += task_names_list[task_index] + " Val Acc:{:.4f} | Val Loss: {:.4f} | Train Acc: {:.4f} | Train Loss: {:.4f}\t".format(accuracies[task_index], val_loss[task_index], accuracies_train[task_index], losses[task_index]) 
                                print(epoch_text)
                                #print('Val Acc:{:.4f}| Val Loss: ASD {:.4f}, ID {:.4f}| Train Acc: {:.4f}'.format(np.mean([acc1,acc2]),valLoss[0],valLoss[1], np.mean([accTrain1,accTrain2])))
                            # Early Stop Checks 
                            
                            
                            if early_stop_enabled:
                                all_fit = True
                                for task_index in range(self.task_count):
                                    if val_loss[task_index] < old_loss[task_index]: 
                                        early_stop_counts[task_index] = 0
                                        old_loss[task_index] = val_loss[task_index]
                                    else:
                                        early_stop_counts[task_index] += 1
                                        
                                    if early_stop_counts[task_index] >= early_stop_window:
                                        if branch_fit[task_index]:
                                            model.branches[task_index].apply(freeze_layer)
                                            if self.task_count > 1:
                                                model.commonmlp.apply(freeze_layer)
                                                optimizers[-1] = torch.optim.Adam(model.commonmlp.parameters(), lr =0.0, weight_decay = wd)
                                            optimizers[task_index] = torch.optim.Adam(model.branches[task_index].parameters(), lr= 0.0, weight_decay = wd)
                                                
                                            print(task_names_list[task_index], ' freezed! Val Loss:{:.4f}'.format(val_loss[task_index]))
                                            early_stop_counts[task_index] = float("-inf")                            
                                        else:
                                            early_stop_counts[task_index] = 0
                                            print("Epoch:", epoch,", ", task_names_list[task_index],"Training Loss:",losses[task_index].mean().item())
                                            print(task_names_list[task_index]," slowed down!")
                                            optimizers[task_index] = torch.optim.Adam(model.branches[task_index].parameters(), lr = l_rate.copy()[task_index] / 20.0, weight_decay = wd)
                                            if self.task_count > 1:
                                                optimizers[-1] = torch.optim.Adam(model.commonmlp.parameters(), lr=l_rate.copy()[-1]/20.0, weight_decay=wd )
                                            branch_fit[task_index] = True
                                    all_fit = all_fit and branch_fit[task_index]
                                
                                if all_fit: 
                                    print("Epoch:", epoch)
                                    for task_index in range(self.task_count):
                                        print(task_names_list[task_index]," Loss:",losses[task_index].mean().item())
                                        # Saving the model with the recommended method on "https://pytorch.org/tutorials/beginner/saving_loading_models.html"
                                        torch.save(model.state_dict(), root + disordername + "Exp" + str(experiment) + "/deepND_trial"+str(j+1)+"_fold"+str(k1+1)+"_"+str(k2+1)+".pth")    
                                    print("Training Done!")
                                    if self.task_count > 1:
                                        model.commonmlp.apply(unfreeze_layer)
                                        for task_index in range(self.task_count):
                                            model.branches[task_index].apply(unfreeze_layer)
                                    elif self.task_count == 1:
                                        model.branches[0].apply(unfreeze_layer)
                                    break
                            
                    # -------------------------------------------------------------
                    
                    mean_predictions = []
                    for task_index in range(self.task_count):
                        #print(output[0][1].cpu()[test_masks[0],1])
                        mean_predictions.append(output[task_index][1][:,1].clone().detach())
                        mean_predictions[task_index][training_masks[task_index]] = 0.0
                        mean_predictions[task_index][left_out_validation_masks[task_index]] = 0.0
                        mean_predictions[task_index] = mean_predictions[task_index].detach().cpu() / (k * (k - 1) * trial)
                        mean_predictions[task_index][left_out_folds[task_index]] *= k
                        predictions[task_index] += mean_predictions[task_index].detach().cpu()
                        #print(output[0][1].cpu()[test_masks[0],1])
                        #exit(0)
                    
                        
                    # -------------------------------------------------------------
                    area_under_rocs = []
                    area_under_prcs = []
                    mcc_values = []
                    for task_index in range(self.task_count):
                        area_under_rocs.append(roc_auc_score(labels[task_index].cpu()[test_masks[task_index]], output[task_index][1].cpu()[test_masks[task_index],1]))
                        aucs[task_index].append(area_under_rocs[task_index])
                        print(task_names_list[task_index], " AUC:", aucs[task_index][-1])
                        
                        
                        
                        area_under_prcs.append(average_precision_score(labels[task_index].cpu()[test_masks[task_index]], output[task_index][1].cpu()[test_masks[task_index],1]))
                        auprcs[task_index].append(area_under_prcs[task_index])
                        print(task_names_list[task_index], " AUPRC:", auprcs[task_index][-1])
                        
                        _, pred =  output[task_index][0].cpu()[test_masks[task_index]].max(dim=1)
                        mcc_values.append(matthews_corrcoef(labels[task_index].cpu()[test_masks[task_index]], pred))
                        mccs[task_index].append(mcc_values[task_index])
                        print(task_names_list[task_index], " MCC:", mccs[task_index][-1])
                        
                        expert_weights = model.branches[task_index].expert_weights.clone().detach()  
                        expert_weights[training_masks[task_index]] = 0.0
                        expert_weights[left_out_validation_masks[task_index]] = 0.0
                        expert_weights = expert_weights.detach().cpu() / (k * (k - 1) * trial)
                        expert_weights[left_out_folds[task_index]] *= k
                        average_expert_weights[task_index] += expert_weights.detach().cpu()
                                
                        expert_probabilities = model.branches[task_index].expert_probabilities.copy()
                        expert_probabilities_merged = torch.zeros((self.instance_count,self.network_count), dtype= torch.float)
                        for i in range(self.network_count):
                            expert_probabilities_merged[:,i] = expert_probabilities[i][:,1]
                        expert_probabilities_merged[training_masks[task_index]] = 0.0
                        expert_probabilities_merged[left_out_validation_masks[task_index]] = 0.0
                        expert_probabilities_merged  = expert_probabilities_merged.detach().cpu() / (k * (k - 1) * trial)
                        expert_probabilities_merged[left_out_folds[task_index]] *= k
                        average_expert_probabilities[task_index] += expert_probabilities_merged
                    # -------------------------------------------------------------
                    print("."*10)
                    for task_index in range(self.task_count):
                        print(task_names_list[task_index], " current median AUC:" + str(np.median(aucs[task_index])))
                        print(task_names_list[task_index], " current median AUPR:" + str(np.median(auprcs[task_index])))
                        print(task_names_list[task_index], " current median MCC:" + str(np.median(mccs[task_index])))
                    print("-"*10)
                
            # -------------------------------------------------------------
            for task_index in range(self.task_count): 
                print(task_names_list[task_index]," trial median AUC :" + str( np.median( aucs[task_index][- (k*(k-1)) :] )))
                print(task_names_list[task_index], " trial median AUPRC:" + str( np.median( auprcs[task_index][- (k*(k-1)) :] )))
                print(task_names_list[task_index], " trial median MCC:" + str( np.median( mccs[task_index][- (k*(k-1)) :] )))                 

            print("-"*80)
        ###############################################################################################################################################    
        """Writing Final Result of the Session"""
        ###############################################################################################################################################
        for task_index in range(self.task_count):
            write_prediction(predictions[task_index], gene_indices[task_index][0], gene_indices[task_index][1], gene_indices[task_index][2], gene_indices[task_index][-1], feature_names, root, task_names_list[task_index], trial, k, experiment, self.disordername)
        
        #Experiment Stats
        write_experiment_stats(root, aucs, auprcs , mccs, disordername, trial, k, self.init_time , self.network_count, mode, task_names_list, self.experiment)

        for i in range(self.task_count):
            torch.save(average_expert_probabilities[i],root + disordername + "Exp" + str(experiment) + "/" + task_names_list[i].lower() + "ExpertProbabilities.pt")
        for i in range(self.task_count):
            torch.save(average_expert_weights[i], root + disordername + "Exp" + str(experiment) + "/" + task_names_list[i].lower() + "ExpertWeights.pt")
        #heatmap7 = all_att_id
        #torch.save(heatmap7, path + "/heatmap_flat_all_id.pt");
        
        #heatmap8 = pre_att_id
        #torch.save(heatmap8, path + "/heatmap_pre_att_id.pt");
    
