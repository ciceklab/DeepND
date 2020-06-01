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
