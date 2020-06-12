"""
main.py
Main Segment of  DeepND 
Bilkent University, Department of Computer Engineering
Ankara, 2020
"""
from models import *
from utils import *
from deepnd_st import *
from deepnd import *
import torch

import sys
import pickle
import numpy as np
import pandas as pd
import os 
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

root = "" # Current directory
trial = 10 # Number of trials to train | Default : 10
k = 5 # k-fold cross validation | Default : 5
mode = 0 # 1 : Test, 0: Train | Default : 0
model_select = 1 # 1 : Multi, 0: Single | Default : 1
disease = 0 # Required for Single Task Mode, 0 : ASD, 1 : ID | Default : 0
networks = [11] # List that contains regions to be fed to the model, example is set for region 11 (temporal window 12-14) | Default (all regions) : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# GPU Mask Setup
# GPU mask MUST have the same length as networks!
pfcgpumask = [0]
mdcbcgpumask = [0]
v1cgpumask = [0]
shagpumask = [0]

# GPU Device Setup
devices = []
for i in range(torch.cuda.device_count()):
    devices.append(torch.device('cuda:' + str(i)))
    
print("CUDA Device Count:",torch.cuda.device_count())
maskCheck([pfcgpumask,mdcbcgpumask,v1cgpumask,shagpumask,networks])    
if model_select:
    diseasename = "Multi"
else:
    if disease:
        diseasename = "ID"
    else:
        diseasename = "ASD"

access_rights = 0o755  # User :RWX | Group : RX | Others : RX    
if mode:
    # Test Mode Directory Setup
    experiment = 0
    if experiment < 10:
        experiment = "0" + str(experiment)
    print("Generating results for ", diseasename , " Exp :", experiment)
    path = root + diseasename + "Exp" + str(experiment) + "Test"
    try:
        os.mkdir(path, access_rights)
    except OSError:
        print ("Creation of the test directory failed. Possibly, the directory already exists.")
    else:
        print ("Successfully created the test directory.")
    # Load random states for reproducing test results
    torch.set_rng_state(torch.load(root + diseasename + "Exp" + str(experiment) + "/deepND_experiment_torch_random_state"))
    state = np.random.get_state()
    with open(root + diseasename + "Exp" + str(experiment) + "/deepND_experiment_numpy_random_state", 'rb') as f:
        state = pickle.load(f)
    np.random.set_state(state)
else:
    # Train Mode Directory Setup
    experiment = 0
    if experiment < 10:
        experiment = "0" + str(experiment)
    print("Training ", diseasename , " for Exp :", experiment)
    path = root + diseasename + "Exp" + str(experiment)
    try:
        os.mkdir(path, access_rights)
    except OSError:
        print ("Creation of the directory for the results failed. Possibly, the directory already exists.")
    else:
        print ("Successfully created the directory for the results.")
    # Save random states for reproducing test results in future
    torch.save(torch.get_rng_state(),root + diseasename + "Exp" + str(experiment) + "/deepND_experiment_torch_random_state")
    state = np.random.get_state()
    with open(root + diseasename + "Exp" + str(experiment) + "/deepND_experiment_numpy_random_state", 'wb') as f:
        pickle.dump(state, f)
        
if model_select:
    # Multi Task Model
    input_size = [29, 17, 13] # featsize = 29 | featsizeasd = 17 | featsizeid = 13 
    l_rate = [0.0007, 0.0007, 0.777] # lrc = 0.0007 | lrasd = 0.0007 | lrid = 0.007
    wd = 0.0001
    diseasename = "Multi"
    deepnd(root, path, input_size, mode,  l_rate, wd, trial, k, diseasename, devices, pfcgpumask, mdcbcgpumask, shagpumask, v1cgpumask, state, experiment, networks)
else:
    # Single Task Model
    if disease:
        input_size = 13
        l_rate = 0.0007  
        diseasename = "ID"
    else:
        input_size = 17
        l_rate = 0.0007 
        diseasename = "ASD"
    deepnd_st( root, path, input_size, mode, l_rate, trial, k, diseasename, devices, pfcgpumask, mdcbcgpumask, shagpumask, v1cgpumask, state, experiment, networks)