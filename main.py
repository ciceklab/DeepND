"""
main.py
Main Segment of  DeepND 
Bilkent University, Department of Computer Engineering
Ankara, 2020
"""
from models import *
from utils import *
from deepnd_st import *
import torch

import sys
import pickle
import numpy as np
import pandas as pd
import os 
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3,4,5,6,7"

root = ""
trial = 10
k = 5
mode = 1
model_select = 0
disease = 0

pfcgpumask = [0,0,0,0,0,0,0,0,0,0,0,0,0]
mdcbcgpumask = [1,1,1,1,1,2,2,2,3,3,3,3,3]
v1cgpumask = [4,4,4,4,4,5,5,5,5,5,5,5,5]
shagpumask = [5,5,6,6,6,6,6,6,6,6,6,6,6]

devices = []
for i in range(torch.cuda.device_count()):
    devices.append(torch.device('cuda:' + str(i)))
    
print("CUDA Device Count:",torch.cuda.device_count())
    
if model_select:
    diseasename = "Multi"
else:
    if disease:
        diseasename = "ID"
    else:
        diseasename = "ASD"

access_rights = 0o755
       
if mode:
    experiment = 0
    if experiment < 10:
        experiment = "0" + str(experiment)
    print("Generating results for ", diseasename , " Exp :", experiment)
    path = root + diseasename + "Exp" + str(experiment) + "Test"
    try:
        os.mkdir(path, access_rights)
    except OSError:
        print ("Creation of the test directory failed")
    else:
        print ("Successfully created the test directory")
    # Load random states for reproducing test results
    torch.set_rng_state(torch.load(root + diseasename + "Exp" + str(experiment) + "/deepND_experiment_torch_random_state"))
    state = np.random.get_state()
    with open(root + diseasename + "Exp" + str(experiment) + "/deepND_experiment_numpy_random_state", 'rb') as f:
        state = pickle.load(f)
    np.random.set_state(state)
else:
    path = root + diseasename + "Exp" + str(experiment)
    try:
        os.mkdir(path, access_rights)
    except OSError:
        print ("Creation of the directory for the results failed")
    else:
        print ("Successfully created the directory for the results")
    torch.save(torch.get_rng_state(),root + diseasename + "Exp" + str(experiment) + "/deepND_experiment_torch_random_state")
    state = np.random.get_state()
    with open(root + diseasename + "Exp" + str(experiment) + "/deepND_experiment_numpy_random_state", 'wb') as f:
        pickle.dump(state, f)
        
if model_select:
    featsizeid = 13 
    featsizeasd = 17
    featsize = 29 
    lrasd = 0.0007
    lrid = 0.007
    lrc = 0.0007
    wd = 0.0001
    diseasename = "Multi"
    model = DeepND()
    # deepnd()
else:
    if disease:
        input_size = 13
        l_rate = 0.0007  
        diseasename = "ID"
    else:
        input_size = 17
        l_rate = 0.0007 
        diseasename = "ASD"
    #model = DeepND_ST(featsize=input_size)
    deepnd_st( root, path, input_size, mode, trial, k, diseasename, devices, pfcgpumask, mdcbcgpumask, shagpumask, v1cgpumask, state)