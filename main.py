"""
main.py
Main Segment of  DeepND 
Bilkent University, Department of Computer Engineering
Ankara, 2020
"""


import sys
import pickle
import numpy as np
import pandas as pd
import os 


# Default parameter settings
root = "" # Current directory
trial = 10 # Number of trials to train | Default : 10
k = 5 # k-fold cross validation | Default : 5
mode = 0 # 1 : Test, 0: Train | Default : 0
experiment = 0 # Experiment ID
model_select = 1 # 1 : Multi, 0: Single | Default : 1
#disorder = 0 # Required for Single Task Mode, 0 : ASD, 1 : ID | Default : 0
#networks = [11] # List that contains regions to be fed to the model, example is set for region 11 (temporal window 12-14) | Default (all regions) : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
networks = "brainspan_all"
feature_sets = "ASD, ID"
moe_feature_sets = None
positive_ground_truths = "ASD, ID"
negative_ground_truths = "ASD, ID"
verbose = 0
network_gpu_mask = "auto"
system_gpu_mask = "0"
l_rate = 0.007
wd = 0.001
hidden_units = 4
disordername = "Multi"
common_layer_units = 15
task_names = "indices"
feature_names = "indices"
torch_seed_value = "random"
numpy_seed_value = "random"

print("Parsing config file.")
filepath = 'config'
with open(filepath) as fp:
   for cnt, line in enumerate(fp):
    if '#' not in line and line.strip() != "": # Ignore lines with a '#' character and blank lines.
       splitted_line = line.split("=")
       
       if len(splitted_line) != 2:
        print("Invalid parameter format at line ", cnt + 1, ", using default value for this parameter.")
        continue
       
       parameter_name = splitted_line[0].strip()
       parameter_value = splitted_line[1].strip()
       
       if parameter_name == "trial_count":
        trial = int(parameter_value)
        
       elif parameter_name == "fold_count":
        k = int(parameter_value)
        
       elif parameter_name == "test_mode":
        mode = int(parameter_value)
        
       elif parameter_name == "verbose":
        verbose = int(parameter_value)
        
       elif parameter_name == "experiment_id":
        experiment = int(parameter_value)
        
       elif parameter_name == "networks":
        networks = parameter_value
        
       elif parameter_name == "feature_sets":
        feature_sets = parameter_value
        feature_sets = create_feature_set_list(feature_sets)
       
       elif parameter_name == "moe_feature_sets":
        moe_feature_sets = parameter_value
        moe_feature_sets = create_feature_set_list(moe_feature_sets)
       
       elif parameter_name == "positive_ground_truths":
        positive_ground_truths = parameter_value
        
       elif parameter_name == "negative_ground_truths":
        negative_ground_truths = parameter_value
        
       elif parameter_name == "system_gpu_mask":
        system_gpu_mask = parameter_value
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]= system_gpu_mask
        from utils import *
        
       elif parameter_name == "network_gpu_mask":
        network_gpu_mask = parameter_value
        
       elif parameter_name == "weight_decay":
        wd = float(parameter_value)
        
       elif parameter_name == "learning_rate":
        l_rate = parameter_value
        l_rates = l_rate.split(',')
        learning_rates = []
        for token in l_rates:
            learning_rates.append(float(token.strip()))
            
       elif parameter_name == "weight_decay":
        wd = float(parameter_value)
        
       elif parameter_name == "hidden_units":
        hidden_units = int(parameter_value)
        
       elif parameter_name == "experiment_name":
        disordername = parameter_value
        
       elif parameter_name == "common_layer_units":
        common_layer_units = int(parameter_value)
       
       elif parameter_name == "task_names":
        task_names = parameter_value
        
       elif parameter_name == "feature_names":
        feature_names = parameter_value
        
       elif parameter_name == "torch_seed":
        if parameter_value.isnumeric():
            torch_seed_value = int(parameter_value)
       elif parameter_name == "numpy_seed":
        if parameter_value.isnumeric():
            numpy_seed_value = int(parameter_value)
        
       
print("Config file has been parsed.")
import torch
if torch_seed_value != "random":
    torch.manual_seed(torch_seed_value)
if numpy_seed_value != "random":
    np.random.seed(numpy_seed_value)
from models import *
from deepnd import *


devices = []
for i in range(torch.cuda.device_count()):
    devices.append(torch.device('cuda:' + str(i)))
  
if verbose:  
    print("CUDA Device Count:",torch.cuda.device_count())
    
'''  
if model_select:
    disordername = "Multi"
else:
    if disease:
        disordername = "ID"
    else:
        disordername = "ASD"
'''     
if experiment < 10:
    experiment = "0" + str(experiment)
        
access_rights = 0o755  # User :RWX | Group : RX | Others : RX    
if mode:
    if verbose:
        print("Test mode activated.")
    # Test Mode Directory Setup
    print("Generating results for ", disordername , " Exp :", experiment)
    path = root + disordername + "Exp" + str(experiment) + "Test"
    try:
        os.mkdir(path, access_rights)
    except OSError:
        if verbose:
            print ("Creation of the test directory failed. Possibly, the directory already exists.")
    else:
        if verbose:
            print ("Successfully created the test directory.")
    # Load random states for reproducing test results
    torch.set_rng_state(torch.load(root + disordername + "Exp" + str(experiment) + "/deepND_experiment_torch_random_state"))
    state = np.random.get_state()
    with open(root + disordername + "Exp" + str(experiment) + "/deepND_experiment_numpy_random_state", 'rb') as f:
        state = pickle.load(f)
    np.random.set_state(state)
else:
    if verbose:
        print("Train mode activated.")
    # Train Mode Directory Setup
    print("Training ", disordername , " for Exp :", experiment)
    path = root + disordername + "Exp" + str(experiment)
    try:
        os.mkdir(path, access_rights)
    except OSError:
        if verbose:
            print ("Creation of the directory for the results failed. Possibly, the directory already exists.")
    else:
        if verbose:
            print ("Successfully created the directory for the results.")
    # Save random states for reproducing test results in future
    torch.save(torch.get_rng_state(),root + disordername + "Exp" + str(experiment) + "/deepND_experiment_torch_random_state")
    state = np.random.get_state()
    with open(root + disordername + "Exp" + str(experiment) + "/deepND_experiment_numpy_random_state", 'wb') as f:
        pickle.dump(state, f)
 
if verbose:
    print("Reading all feature sets.")
features = load_all_features(feature_sets)
moe_features = load_all_features(moe_feature_sets)

input_size = []
for feature in features:
    input_size.append(feature.shape[1])
    #input_size.append(1)
    
moe_feat_size = []

for moe_feature in moe_features:
    if moe_feature.dim() == 1:
        moe_feat_size.append(1)
    else:
        moe_feat_size.append(moe_feature.shape[1])


if verbose:
    print("All features have been read and processed.\n")

if verbose:
    print("Reading network tensor files.\n")
networks = create_network_list(networks)
driver = DeepND_Driver(root,
         input_size,
         mode,
         learning_rates,
         wd,
         hidden_units,
         trial,
         k,
         disordername,
         devices,
         network_gpu_mask,
         state,
         experiment,
         networks,
         positive_ground_truths,
         negative_ground_truths,
         features,
         verbose,
         system_gpu_mask,
         network_gpu_mask,
         common_layer_units,
         task_names,
         feature_names,
         moe_features,
         moe_feat_size)

