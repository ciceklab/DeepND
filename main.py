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

root = "/" # PATH
mode = 0 # 0 : Train | 1 : Test
model = 0 # 0 : Single | 1 : Multi
disease = 0 # 0 : ASD | 1 : ID

network_count = 13
max_epoch = 1000
early_stop_enabled = 1
early_stop_window = 7
access_rights = 0o755

if model:
    featsizeid = 13 
    featsizeasd = 18
    featsize = 28 
    h1 = 4
    h2 = 2
    lrasd=0.001
    lrid=0.002
    lrc=0.001
    wd=0.0001
    diseasename = "Multi"
else:
    if disease:
        input_size = 13
        l_rate = 0.0007  
        diseasename = "ID"
    else:
        input_size = 18
        l_rate = 0.0007 
        diseasename = "ASD"
  
if mode:
    experiment = 11
    if experiment < 10:
        experiment = "0"+str(experiment)
    print("Generating results for ", diseasename," Exp :", experiment)
    try:
        os.mkdir(root+diseasename+"Exp"+str(experiment)+"Test", access_rights)
    except OSError:
        print ("Creation of the test directory failed")
    else:
        print ("Successfully created the test directory")
    # Load random states for reproducing test results
    torch.set_rng_state(torch.load(root+diseasename+"Exp"+str(experiment)+"/deepND_experiment_torch_random_state"))
    state =np.random.get_state()
    with open(root+diseasename+"Exp"+str(experiment)+"/deepND_experiment_numpy_random_state", 'rb') as f:
        state = pickle.load(f)
    np.random.set_state(state)
else:
    try:
        os.mkdir(root+diseasename+"Exp"+str(experiment), access_rights)
    except OSError:
        print ("Creation of the directory for the results failed")
    else:
        print ("Successfully created the directory for the results")
    torch.save(torch.get_rng_state(),root+diseasename+"Exp"+str(experiment)+"/deepND_experiment_torch_random_state")
    state =np.random.get_state()
    with open(root+diseasename+"Exp"+str(experiment)+"/deepND_experiment_numpy_random_state", 'wb') as f:
        pickle.dump(state, f)
    
