"""
deepnd.py
Training and test processes of DeepND
for replicating previous experiments and reproducing data
Bilkent University, Department of Computer Engineering
Ankara, 2020
"""
###############################################################################################################################################
"""GOLD STANDARDS"""
###############################################################################################################################################
pos_gold_standards_asd = pd.read_csv(root + "/Data/ASD_Pos_Gold_Standards.csv")
neg_gold_standards_asd = pd.read_csv(root + "/Data/ASD_Neg_Gold_Standards.csv")

pos_gold_std_asd = pos_gold_standards_asd.values
neg_gold_std_asd = neg_gold_standards_asd.values

pos_gold_std_genes_asd = [str(item) for item in pos_gold_std_asd[:,0]]
pos_gold_std_evidence_asd = [str(item) for item in pos_gold_std_asd[:,2]]
neg_gold_std_genes_asd = [str(item) for item in neg_gold_std_asd[:,0]]

y1 = torch.zeros(len(geneNames_all), dtype = torch.long)

pgold_tada_intersect_asd, pgold_indices_asd, pgold_delete_indices_asd, g_bs_tada_intersect_indices_asd = intersect_lists(pos_gold_std_genes_asd , [str(item) for item in geneNames_all], geneDict)
ngold_tada_intersect_asd, ngold_indices_asd, ngold_delete_indices_asd, n_bs_tada_intersect_indices_asd = intersect_lists(neg_gold_std_genes_asd , [str(item) for item in geneNames_all], geneDict)
y1[g_bs_tada_intersect_indices_asd] = 1
y1[n_bs_tada_intersect_indices_asd] = 0

gold_evidence_asd = [pos_gold_std_evidence_asd[item] for item in pgold_indices_asd]

print("\n", len(pgold_tada_intersect_asd), " Many Positive ASD Gold Standard Genes are Found!")
print(len([pos_gold_std_genes_asd[item] for item in pgold_delete_indices_asd]), " Many Positive ASD Gold Standard Genes Cannot be Found!")
print("\n", len(ngold_tada_intersect_asd), " Many Negative ASD Gold Standard Genes are Found!")
print(len([neg_gold_std_genes_asd[item] for item in ngold_delete_indices_asd]), " Many Negative ASD Gold Standard Genes Cannot be Found!")

pos_gold_standards_id = pd.read_csv(root + "/Data/ID_Pos_Gold_Standards.csv")
neg_gold_standards_id = pd.read_csv(root + "/Data/ID_Neg_Gold_Standards.csv")

pos_gold_std_id = pos_gold_standards_id.values
neg_gold_std_id = neg_gold_standards_id.values

pos_gold_std_genes_id = [str(item) for item in pos_gold_std_id[:,0]]
pos_gold_std_evidence_id = [str(item) for item in pos_gold_std_id[:,2]]
neg_gold_std_genes_id = [str(item) for item in neg_gold_std_id[:,0]]

y2 = torch.zeros(len(geneNames_all), dtype = torch.long)

pgold_tada_intersect_id, pgold_indices_id, pgold_delete_indices_id, g_bs_tada_intersect_indices_id = intersect_lists(pos_gold_std_genes_id , [str(item) for item in geneNames_all], geneDict)
ngold_tada_intersect_id, ngold_indices_id, ngold_delete_indices_id, n_bs_tada_intersect_indices_id = intersect_lists(neg_gold_std_genes_id , [str(item) for item in geneNames_all], geneDict)
y2[g_bs_tada_intersect_indices_id] = 1
y2[n_bs_tada_intersect_indices_id] = 0

gold_evidence_id = [pos_gold_std_evidence_id[item] for item in pgold_indices_id]

print("\n", len(pgold_tada_intersect_id), " Many Positive ID Gold Standard Genes are Found!")
print(len([pos_gold_std_genes_id[item] for item in pgold_delete_indices_id]), " Many Positive ID Gold Standard Genes Cannot be Found!")
print("\n", len(ngold_tada_intersect_id), " Many Negative ID Gold Standard Genes are Found!")
print(len([neg_gold_std_genes_id[item] for item in ngold_delete_indices_id]), " Many Negative ID Gold Standard Genes Cannot be Found!")

###############################################################################################################################################
"""EVIDENCE AND VALIDATION SETS"""
###############################################################################################################################################
k = 5 # k for k-fold cross validation with k-2/1/1 validation
# If another validation set is used, gene counts must be updated. This part could be done automatically as well by checking gene evidences and standard values from files
asd_e1_gene_count = 0
asd_e2_gene_count = 0
asd_e3e4_gene_count = 0
asd_e1_gene_indices = []
asd_e2_gene_indices = []
asd_e3e4_gene_indices = []
for index,i in enumerate(gold_evidence_asd):
    if i == "E1":
        asd_e1_gene_count += 1
        asd_e1_gene_indices.append(g_bs_tada_intersect_indices_asd[index])
    elif i == "E2":
        asd_e2_gene_count += 1
        asd_e2_gene_indices.append(g_bs_tada_intersect_indices_asd[index])
    else:
        asd_e3e4_gene_count += 1
        asd_e3e4_gene_indices.append(g_bs_tada_intersect_indices_asd[index])
        
asd_e1_fold_size = math.ceil(asd_e1_gene_count / k)
asd_e2_fold_size = math.ceil(asd_e2_gene_count / k)
asd_e3e4_fold_size = math.ceil(asd_e3e4_gene_count / k)
asd_neg_gene_count = len(n_bs_tada_intersect_indices_asd)
asd_neg_fold_size = math.ceil(asd_neg_gene_count / k)

print("ASD E1 Gene Count:", asd_e1_gene_count)
print("ASD E2 Gene Count:", asd_e2_gene_count)
print("ASD E3E4 Gene Count:", asd_e3e4_gene_count)

np.random.set_state(state)
asd_e1_perm = np.random.permutation(asd_e1_gene_count)
asd_e2_perm = np.random.permutation(asd_e2_gene_count)
asd_e3e4_perm = np.random.permutation(asd_e3e4_gene_count)
asd_neg_perm = np.random.permutation(asd_neg_gene_count)    

id_e1_gene_count = 0
id_e2_gene_count = 0
id_e3e4_gene_count = 0
id_e1_gene_indices = []
id_e2_gene_indices = []
id_e3e4_gene_indices = []
for index,i in enumerate(gold_evidence_id):
    if i == "E1":
        id_e1_gene_count += 1
        id_e1_gene_indices.append(g_bs_tada_intersect_indices_id[index])
        
    elif i == "E2":
        id_e2_gene_count += 1
        id_e2_gene_indices.append(g_bs_tada_intersect_indices_id[index])
    else:
        id_e3e4_gene_count += 1
        id_e3e4_gene_indices.append(g_bs_tada_intersect_indices_id[index])
        
id_e1_fold_size = math.ceil(id_e1_gene_count / k)
id_e2_fold_size = math.ceil(id_e2_gene_count / k)
id_e3e4_fold_size = math.ceil(id_e3e4_gene_count / k)
id_neg_gene_count = len(n_bs_tada_intersect_indices_id)
id_neg_fold_size = math.ceil(id_neg_gene_count / k)

print("ID E1 Gene Count:", id_e1_gene_count)
print("ID E2 Gene Count:", id_e2_gene_count)
print("ID E3E4 Gene Count:", id_e3e4_gene_count)

np.random.set_state(state)
id_e1_perm = np.random.permutation(id_e1_gene_count)  
id_e2_perm = np.random.permutation(id_e2_gene_count)
id_e3e4_perm = np.random.permutation(id_e3e4_gene_count)
id_neg_perm = np.random.permutation(id_neg_gene_count)
###############################################################################################################################################
"""FEATURES"""
###############################################################################################################################################

featuresasd = np.load(root + "/Data/ASD_TADA_Features.npy")
featuresasd = torch.from_numpy(featuresasd).float()
featuresasd = (featuresasd - torch.mean(featuresasd,0)) / (torch.std(featuresasd,0))

data_asd = Data(x=featuresasd)
featuresasd = featuresasd.to(devices[0])
data_asd = data_asd.to(devices[0])
data_asd.y1 = y1.to(devices[0])

featuresid = np.load(root + "/Data/ID_TADA_Features.npy")
featuresid = torch.from_numpy(featuresid).float()
featuresid = (featuresid - torch.mean(featuresid,0)) / (torch.std(featuresid,0))

data_id = Data(x=featuresid)
featuresid = featuresid.to(devices[0])
data_id = data_id.to(devices[0])
data_id.y2 = y2.to(devices[0])

gene_names_list = [str(item) for item in geneNames_all]

commonfeatures = np.load(root + "/Data/Common_TADA_Features.npy")
commonfeatures = torch.from_numpy(commonfeatures).float()
commonfeatures = (commonfeatures - torch.mean(commonfeatures,0)) / (torch.std(commonfeatures,0))

commonfeatures = Data(x=commonfeatures)
features = commonfeatures.x.to(devices[0]) 

###############################################################################################################################################    
"""MODEL CONSTRUCTION"""
###############################################################################################################################################
model = DeepND()
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
aupr_id = []
aucs_id = []  
usage = 0
cached = 0

trial = 10
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
        test_mask_asd = [asd_e1_gene_indices[index] for index in asd_e1_perm[k1 * asd_e1_fold_size: min(asd_e1_gene_count, (k1 + 1) * asd_e1_fold_size) ] ]
        test_mask_asd +=  [n_bs_tada_intersect_indices_asd[item] for item in asd_neg_perm[k1 * asd_neg_fold_size : min(asd_neg_gene_count , (k1 + 1) * asd_neg_fold_size)] ]
        data_asd.test_mask = test_mask_asd.copy()
        
        test_mask_asd += [asd_e2_gene_indices[index] for index in asd_e2_perm[(k1) * asd_e2_fold_size: min(asd_e2_gene_count, (k1 + 1) * asd_e2_fold_size) ] ] 
        test_mask_asd += [asd_e3e4_gene_indices[index] for index in asd_e3e4_perm[(k1) * asd_e3e4_fold_size: min(asd_e3e4_gene_count, (k1 + 1) * asd_e3e4_fold_size) ] ] 
        
        asd_k_e1_perm = np.delete(asd_e1_perm,np.s_[k1*asd_e1_fold_size:min(asd_e1_gene_count,(k1 + 1) * asd_e1_fold_size)],axis=0)
        asd_k_neg_perm = np.delete(asd_neg_perm,np.s_[k1 * asd_neg_fold_size: min(asd_neg_gene_count, (k1 + 1) * asd_neg_fold_size) ],axis=0)
        asd_k_e2_perm = np.delete(asd_e2_perm,np.s_[k1 * asd_e2_fold_size: min(asd_e2_gene_count, (k1 + 1) * asd_e2_fold_size) ],axis=0)
        asd_k_e3e4_perm = np.delete(asd_e3e4_perm,np.s_[k1 * asd_e3e4_fold_size: min(asd_e3e4_gene_count, (k1 + 1) * asd_e3e4_fold_size) ],axis=0)
        
        #ID
        test_mask_id = [id_e1_gene_indices[index] for index in id_e1_perm[k1 * id_e1_fold_size: min(id_e1_gene_count, (k1 + 1) * id_e1_fold_size) ] ]
        test_mask_id +=  [n_bs_tada_intersect_indices_id[item] for item in id_neg_perm[k1 * id_neg_fold_size : min(id_neg_gene_count , (k1 + 1) * id_neg_fold_size)] ]
        data_id.test_mask = test_mask_id.copy()
        
        test_mask_id += [id_e2_gene_indices[index] for index in id_e2_perm[(k1) * id_e2_fold_size: min(id_e2_gene_count, (k1 + 1) * id_e2_fold_size) ] ] 
        test_mask_id += [id_e3e4_gene_indices[index] for index in id_e3e4_perm[(k1) * id_e3e4_fold_size: min(id_e3e4_gene_count, (k1 + 1) * id_e3e4_fold_size) ] ] 
        
        id_k_e1_perm = np.delete(id_e1_perm,np.s_[k1*id_e1_fold_size:min(id_e1_gene_count,(k1 + 1) * id_e1_fold_size)],axis=0)
        id_k_neg_perm = np.delete(id_neg_perm,np.s_[k1 * id_neg_fold_size: min(id_neg_gene_count, (k1 + 1) * id_neg_fold_size) ],axis=0)
        id_k_e2_perm = np.delete(id_e2_perm,np.s_[k1 * id_e2_fold_size: min(id_e2_gene_count, (k1 + 1) * id_e2_fold_size) ],axis=0)
        id_k_e3e4_perm = np.delete(id_e3e4_perm,np.s_[k1 * id_e3e4_fold_size: min(id_e3e4_gene_count, (k1 + 1) * id_e3e4_fold_size) ],axis=0)
        
        for k2 in range(k-1): # K-FOLD Cross Validation
            print("Fold", k1+1, "_",  k2+1, "of Trial", j+1)
            
            # Adjust masks - NOTE: Masks contain indices of samples. 
            # Example: if train mask contains 2 genes with indices 6 and 12, train mask should be --> train_mask = [6, 12]
            # Add leftout E1 genes to validation mask
            
            # ASD
            validation_mask_asd = [asd_e1_gene_indices[index] for index in asd_k_e1_perm[(k2) * asd_e1_fold_size: min(asd_e1_gene_count, (k2 + 1) * asd_e1_fold_size) ] ] 
            print("ASD Validation Mask Length After E1:", len(validation_mask_asd))
            print('ASD Validation Gene(s):', [gene_names_list[i] for i in validation_mask_asd])
            # Add negative genes to validation mask
            validation_mask_asd +=  [n_bs_tada_intersect_indices_asd[item] for item in asd_k_neg_perm[k2 * asd_neg_fold_size : min(asd_neg_gene_count , (k2 + 1) * asd_neg_fold_size)] ]
            data_asd.auc_mask = validation_mask_asd.copy()
            #print("Supposed AUC Mask Length:", len(data.auc_mask))
            validation_mask_asd += [asd_e2_gene_indices[index] for index in asd_k_e2_perm[(k2) * asd_e2_fold_size: min(asd_e2_gene_count, (k2 + 1) * asd_e2_fold_size) ] ] 
            validation_mask_asd += [asd_e3e4_gene_indices[index] for index in asd_k_e3e4_perm[(k2) * asd_e3e4_fold_size: min(asd_e3e4_gene_count, (k2 + 1) * asd_e3e4_fold_size) ] ] 
            
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
            validation_mask_id = [id_e1_gene_indices[index] for index in id_k_e1_perm[(k2) * id_e1_fold_size: min(id_e1_gene_count, (k2 + 1) * id_e1_fold_size) ] ] 
            print("ID Validation Mask Length After E1:", len(validation_mask_id))
            print('ID Validation Gene(s):', [gene_names_list[i] for i in validation_mask_id])
            # Add negative genes to validation mask
            validation_mask_id +=  [n_bs_tada_intersect_indices_id[item] for item in id_k_neg_perm[k2 * id_neg_fold_size : min(id_neg_gene_count , (k2 + 1) * id_neg_fold_size)] ]
            data_id.auc_mask = validation_mask_id.copy()
            #print("Supposed AUC Mask Length:", len(data.auc_mask))
            validation_mask_id += [id_e2_gene_indices[index] for index in id_k_e2_perm[(k2) * id_e2_fold_size: min(id_e2_gene_count, (k2 + 1) * id_e2_fold_size) ] ] 
            validation_mask_id += [id_e3e4_gene_indices[index] for index in id_k_e3e4_perm[(k2) * id_e3e4_fold_size: min(id_e3e4_gene_count, (k2 + 1) * id_e3e4_fold_size) ] ] 
            
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
                path = root + diseasename + "Exp" + str(experiment) + "test"
                model.load_state_dict(torch.load(root + diseasename + "Exp" + str(experiment) + "/deepND_trial"+str(j+1)+"_fold"+str(k1+1)+"_"+str(k2+1)+".pth"))
                model = model.eval()
                with torch.no_grad():
                    out1, out2 = model(features, featuresasd, featuresid, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights)
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
                path = root + diseasename + "Exp" + str(experiment)
                model.apply(unfreeze_layer)
                model.apply(weight_reset)
                optimizerc = torch.optim.Adam(model.commonmlp.parameters(), lr = lrc, weight_decay = wd )   
                optimizerasd = torch.optim.Adam( model.ASDBranch.parameters(), lr = lrasd, weight_decay = wd )   
                optimizerid = torch.optim.Adam(model.IDBranch.parameters(), lr = lrid, weight_decay = wd ) 
                
                old_loss = [100,100]
            
                early_stop_count_asd = 0
                early_stop_count_id = 0
                ASDFit = False
                IDFit = False
            
                for epoch in range(max_epoch):
                    model = model.train()
                    optimizerc.zero_grad()
                    optimizerasd.zero_grad()
                    optimizerid.zero_grad()
                    out1,out2 = model(features, featuresasd, featuresid, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights)
                    
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
                        out1, out2 = model(features, featuresasd, featuresid, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights)
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
            fpr["micro"], tpr["micro"], _ = roc_curve(data_asd.y1.cpu()[data_asd.test_mask],(F.softmax(out1.cpu()[data_asd.test_mask, :],dim=1))[:,1])
            aucs_asd.append(auc(fpr["micro"], tpr["micro"]))                                            
            print("ASD AUC", auc(fpr["micro"], tpr["micro"]))
            aupr_asd.append(average_precision_score(data_asd.y1.cpu()[data_asd.test_mask],(F.softmax(out1.cpu()[data_asd.test_mask, :],dim=1))[:,1]))
            print("ASD AUPR", aupr_asd[-1])
            
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
            fpr["micro"], tpr["micro"], _ = roc_curve(data_id.y2.cpu()[data_id.test_mask],(F.softmax(out2.cpu()[data_id.test_mask, :],dim=1))[:,1])
            aucs_id.append(auc(fpr["micro"], tpr["micro"]))                                            
            print("ID AUC", auc(fpr["micro"], tpr["micro"]))
            print("ID AUC", auc(fpr["micro"], tpr["micro"]))
            aupr_id.append(average_precision_score(data_id.y2.cpu()[data_id.test_mask],(F.softmax(out2.cpu()[data_id.test_mask, :],dim=1))[:,1]))
            print("ID AUPR", aupr_id[-1])
            print("."*10)
        
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

            print("ASD Current Median AUC:" + str(np.median(aucs_asd)))
            print("ASD Current Median AUPR:" + str(np.median(aupr_asd)))    
            print("ID Current Median AUC:" + str(np.median(aucs_id)))
            print("ID Current Median AUPR:" + str(np.median(aupr_id))) 
            print("-"*10)
        
    # -------------------------------------------------------------
    print("ASD Trial Mean AUC:" + str(np.mean(aucs_asd[-20:])))
    print("ASD Trial Mean AUPR:" + str(np.mean(aupr_asd[-20:])))    
    print("ID Trial Mean AUC:" + str(np.mean(aucs_id[-20:])))
    print("ID Trial Mean AUPR:" + str(np.mean(aupr_id[-20:])))
    
    print("-"*10)
    print("ASD Current Median AUC:" + str(np.median(aucs_asd)))
    print("ASD Current Median AUPR:" + str(np.median(aupr_asd)))    
    print("ID Current Median AUC:" + str(np.median(aucs_id)))
    print("ID Current Median AUPR:" + str(np.median(aupr_id)))
    print("-"*80)

###############################################################################################################################################    
"""Writing Final Result of the Session"""
###############################################################################################################################################
   
#ASD final Predictions
predictions_asd /= float(trial*k*(k-1))
predictions_asd[g_bs_tada_intersect_indices_asd + n_bs_tada_intersect_indices_asd] *= float(k)

fpred = open(path + "/predictasd.txt","w+")
fpred.write('Probability,Gene Name,Gene ID,Positive Gold Standard,Negative Gold Standard\n')
for index,row in enumerate(predictions_asd):
    if str(geneNames_all[index]) in geneDict:
        fpred.write('%s,%s,%d,%d,%d\n' % (str(row.item()), str(geneDict[str(geneNames_all[index])][0]), geneNames_all[index], 1 if str(geneNames_all[index]) in pos_gold_std_genes_asd else 0, 1 if str(geneNames_all[index]) in neg_gold_std_genes_asd else 0   ) )
    else:
        fpred.write('%s,%s,%d,%d,%d\n' % (str(row.item()), str(geneNames_all[index]), geneNames_all[index], 1 if str(geneNames_all[index]) in pos_gold_std_genes_asd else 0, 1 if str(geneNames_all[index]) in neg_gold_std_genes_asd else 0 ) )
fpred.close()

#ID final Predictions
predictions_id /= float(trial*k*(k-1))
predictions_id[g_bs_tada_intersect_indices_id + n_bs_tada_intersect_indices_id] *= float(k)

fpred = open(path + "/predictid.txt","w+")
fpred.write('Probability,Gene Name,Gene ID,Positive Gold Standard,Negative Gold Standard\n')
for index,row in enumerate(predictions_id):
    if str(geneNames_all[index]) in geneDict:
        fpred.write('%s,%s,%d,%d,%d\n' % (str(row.item()), str(geneDict[str(geneNames_all[index])][0]), geneNames_all[index], 1 if str(geneNames_all[index]) in pos_gold_std_genes_id else 0, 1 if str(geneNames_all[index]) in neg_gold_std_genes_id else 0   ) )
    else:
        fpred.write('%s,%s,%d,%d,%d\n' % (str(row.item()), str(geneNames_all[index]), geneNames_all[index], 1 if str(geneNames_all[index]) in pos_gold_std_genes_id else 0, 1 if str(geneNames_all[index]) in neg_gold_std_genes_id else 0 ) )
fpred.close()

#Experiment Stats
f.write("Number of networks per region: %d\n" % network_count)
print("Number of networks per region:" , network_count)

f.write("\n(ASD) Mean (\u03BC) AUC of All Runs:%f\n" % np.mean(aucs_asd) )
print("(ASD) Mean(\u03BC) AUC of All Runs:", np.mean(aucs_asd) )
f.write("(ASD) \u03C3 of AUCs of All Runs:%f\n" % np.std(aucs_asd) )
print("(ASD) \u03C3 of AUCs of All Runs:", np.std(aucs_asd) )
f.write("(ASD) Median of AUCs of All Runs:%f\n" % np.median(aucs_asd) )
print("(ASD) Meadian of AUCs of All Runs:", np.median(aucs_asd) )

f.write("\n(ASD) Mean (\u03BC) APRC of All Runs:%f\n" % np.mean(aupr_asd) )
print("(ASD) Mean(\u03BC) AUPR of All Runs:", np.mean(aupr_asd) )
f.write("(ASD) \u03C3 of AUPR of All Runs:%f\n" % np.std(aupr_asd) )
print("(ASD) \u03C3 of AUPR of All Runs:", np.std(aupr_asd) )
f.write("(ASD) Median of AUPR of All Runs:%f\n" % np.median(aupr_asd) )
print("(ASD) Meadian of AUCs of All Runs:", np.median(aupr_asd) )

f.write("\n(ID)Mean (\u03BC) AUC of All Runs:%f\n" % np.mean(aucs_id) )
print("(ID)Mean (\u03BC) AUC of All Runs:", np.mean(aucs_id) )
f.write("(ID)\u03C3 of AUCs of All Runs:%f\n" % np.std(aucs_id) )
print("(ID)\u03C3  of AUCs of All Runs:", np.std(aucs_id) )
f.write("(ID)Median of AUCs of All Runs:%f\n" % np.median(aucs_id) )
print("(ID)Meadian of AUCs of All Runs:", np.median(aucs_id) )

f.write("\n(ID)Mean (\u03BC) AUPR of All Runs:%f\n" % np.mean(aupr_id) )
print("(ID)Mean (\u03BC) AUPR of All Runs:", np.mean(aupr_id) )
f.write("(ID)\u03C3 of AUPRs of All Runs:%f\n" % np.std(aupr_id) )
print("(ID)\u03C3  of AUPRs of All Runs:", np.std(aupr_id) )
f.write("(ID)Median of AUPRs of All Runs:%f\n" % np.median(aupr_id) )
print("(ID)Meadian of AUPRs of All Runs:", np.median(aupr_id) )

t = timedelta(seconds=(time.time()-init_time))
f.write("\nDone in %s hh:mm:ss.\n" % t )

f.write("*"*80+"\n") 

for i in range(len(average_attasd)):
    average_attasd[i] = average_attasd[i] / (trial*k*(k-1))
    average_att_goldasd[i] = average_att_goldasd[i] / (trial*k*(k-1))
    stddev_attasd[i] = stddev_attasd[i] / (trial*k)
    stddev_att_goldasd[i] = stddev_att_goldasd[i] / (trial*k*(k-1))
    average_att_gold_e1asd[i] = average_att_gold_e1asd[i] / (trial*k*(k-1))
    average_att_gold_e1e2asd[i] = average_att_gold_e1e2asd[i] / (trial*k*(k-1))
    average_att_gold_negasd[i] = average_att_gold_negasd[i] / (trial*k*(k-1))
    all_att_asd[i] = all_att_asd[i] /(trial*k*(k-1))
    all_att_asd[i][g_bs_tada_intersect_indices_asd + n_bs_tada_intersect_indices_asd] *= 5.0
    pre_att_asd[i] = pre_att_asd[i] /(trial*k*(k-1))
    pre_att_asd[i][g_bs_tada_intersect_indices_asd + n_bs_tada_intersect_indices_asd] *= 5.0

for i in range(network_count * 4):
    print("ASD Average Attention", i + 1 , " of All Runs:", average_attasd[i])
    f.write("Average Attention %d of all runs:%f\n" %( (i + 1), average_attasd[i]))
    f.write("Average Gold Attention %d of all runs:%f\n" %( (i + 1), average_att_goldasd[i]))        
    f.write("Stddev attention %d of all runs:%f\n" %( (i + 1), stddev_attasd[i]))
    f.write("Gold Stddev attention %d of all runs:%f\n\n" %( (i + 1), stddev_att_goldasd[i]))

for i in range(len(average_attid)):
    average_attid[i] = average_attid[i] / (trial*k*(k-1))
    average_att_goldid[i] = average_att_goldid[i] / (trial*k*(k-1))
    stddev_attid[i] = stddev_attid[i] / (trial*k*(k-1))
    stddev_att_goldid[i] = stddev_att_goldid[i] / (trial*k*(k-1))
    average_att_gold_e1id[i] = average_att_gold_e1id[i] / (trial*k*(k-1))
    average_att_gold_e1e2id[i] = average_att_gold_e1e2id[i] / (trial*k*(k-1))
    average_att_gold_negid[i] = average_att_gold_negid[i] / (trial*k*(k-1))
    all_att_id[i] = all_att_id[i] /(trial*k*(k-1))
    all_att_id[i][g_bs_tada_intersect_indices_id + n_bs_tada_intersect_indices_id] *= 5.0
    pre_att_id[i] = pre_att_id[i] /(trial*k*(k-1))
    pre_att_id[i][g_bs_tada_intersect_indices_id + n_bs_tada_intersect_indices_id] *= 5.0

for i in range(network_count * 4):
    print("ID Average Attention", i + 1 , " of All Runs:", average_attid[i])
    f.write("Average Attention %d of all runs:%f\n" %( (i + 1), average_attid[i]))
    f.write("Average Gold Attention %d of all runs:%f\n" %( (i + 1), average_att_goldid[i]))        
    f.write("Stddev attention %d of all runs:%f\n" %( (i + 1), stddev_attid[i]))
    f.write("Gold Stddev attention %d of all runs:%f\n\n" %( (i + 1), stddev_att_goldid[i]))

f.write("*"*80+"\n") 
for i in range(len(aucs_asd)):
    f.write("ASD AUC:%f\n" % (aucs_asd[i]))    
f.write("-"*20+"\n") 
for i in range(len(aupr_asd)):
    f.write("ASD AUPR:%f\n" % (aupr_asd[i]))    
f.write("-"*20+"\n") 
for i in range(len(aucs_id)):
    f.write("ID AUC:%f\n" % (aucs_id[i]))    
f.write("-"*20+"\n") 
for i in range(len(aupr_id)):
    f.write("ID AUPR:%f\n" % (aupr_id[i]))
f.write("-"*20+"\n")
for i in range(len(epoch_count)):
    f.write("ID AUPR:%f\n" % (epoch_count[i]))    

f.close()
print("Generated DeepND results for Exp: ", experiment)
print("Done in ", t , "hh:mm:ss." )

model = model.eval()
out1,out2 = model(features, featuresasd, featuresid, pfcnetworks, mdcbcnetworks, v1cnetworks, shanetworks, pfcnetworkweights, mdcbcnetworkweights, v1cnetworkweights, shanetworkweights)
# ASD
_, pred1 = out1.max(dim=1)
nb_classes = 2
confusion_matrix_asd = torch.zeros(nb_classes, nb_classes,dtype=torch.long)
with torch.no_grad():
    _, preds = torch.max(out1, 1)
    for t, p in zip(data_asd.y1[data_asd.auc_mask].view(-1), preds[data_asd.auc_mask].view(-1)):
        confusion_matrix_asd[t.long(), p.long()] += 1
print("ASD:\n",confusion_matrix_asd)
print(preds.sum().item(), " Many nodes are labeled as ID Genes")
print(F.softmax(out1, dim = 1)[:,1])

# ID
_, pred2 = out2.max(dim=1)
confusion_matrix_id = torch.zeros(nb_classes, nb_classes,dtype=torch.long)
with torch.no_grad():
    _, preds = torch.max(out2, 1)
    for t, p in zip(data_id.y2[data_id.auc_mask].view(-1), preds[data_id.auc_mask].view(-1)):
        confusion_matrix_id[t.long(), p.long()] += 1
print("ID:\n",confusion_matrix_id)
print(preds.sum().item(), " Many nodes are labeled as ID Genes")
print(F.softmax(out2, dim = 1)[:,1])
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
