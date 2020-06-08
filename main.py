"""
main.py
Main Segment of  DeepND 
Bilkent University, Department of Computer Engineering
Ankara, 2020
"""
import models
import utils
import train
import test

import sys
import pickle

import os 
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3,4,5,6,7"

trial = 10
k = 5
mode = 1
model_select = 0
disease = 0

devices = []
for i in range(torch.cuda.device_count()):
    devices.append(torch.device('cuda:' + str(i)))
    
if mode:
    experiment = 1
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
    featsizeasd = 18
    featsize = 28 
    h1 = 4
    h2 = 2
    lrasd = 0.0007
    lrid = 0.007
    lrc = 0.0007
    wd = 0.0001
    diseasename = "Multi"
    model = DeepND()
    deepnd()
else:
    if disease:
        input_size = 13
        l_rate = 0.0007  
        diseasename = "ID"
    else:
        input_size = 18
        l_rate = 0.0007 
        diseasename = "ASD"
    model = DeepND_ST(featsize=input_size)
    deepnd_st( root = root, path = path, mode = 0, trial=trial, k=k, diseasename = diseasename)