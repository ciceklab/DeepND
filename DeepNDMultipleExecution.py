import os
import torch
import pandas as pd
def thread_function(name):
    print("Starting execution with id:", name)
    file1 = open('config', 'r') 
    Lines = file1.readlines() 
    outF = open("nextconfig", "w")
    count = 0
    # Strips the newline character 
    for line in Lines: 
        if(line[0] == '#'):
            outF.write(line)
        elif line == "\n":
            outF.write(line)
        elif "positive_ground_truths" in line:
            outF.write("positive_ground_truths=Data/RandomLabels/RandomPos" + str(name) + ".csv,Data/RandomLabels/RandomPosID" + str(name) + ".csv\n")
        elif "negative_ground_truths" in line:
            outF.write("negative_ground_truths=Data/RandomLabels/RandomNeg" + str(name) + ".csv,Data/RandomLabels/RandomNegID" + str(name) + ".csv\n")
        elif "experiment_id" in line:
            outF.write("experiment_id = " + str(name) + "\n")
        else:
            outF.write(line)
    outF.close()
    os.system("rm config")
    os.system("mv nextconfig config")
    os.system("python3.6 main.py")
    
    
outF = open("random_results.csv", "w")
for i in range(26, 100):
    thread_function(str(i))
    if i < 10:
        name = "0" + str(i)
    else:
        name = str(i)
    heatmap = torch.load("mt_with_identity_newidset_moepli_randomlabelsExp" + name + "/asdExpertProbabilities.pt")
    gold_standards = pd.read_csv("Data/RandomLabels/RandomPos" + str(i) + ".csv").values
    krishnan_genes = pd.read_csv("Data/row-genes.txt").values
    e1_indices = []
    for index,row in enumerate(gold_standards):
        if row[2] == "E1":
            e1_indices.append(krishnan_genes[:,0].tolist().index(row[0]))
    print(len(e1_indices))
    heatmap = torch.mean(heatmap[e1_indices,:], dim = 0)
    
    
    heatmap2 = torch.load("mt_with_identity_newidset_moepli_randomlabelsExp" + name + "/asdExpertWeights.pt")
    gold_standards = pd.read_csv("Data/RandomLabels/RandomPos" + str(i) + ".csv").values
    #krishnan_genes = pd.read_csv("Data/row-genes.txt").values
    e1_indices = []
    for index,row in enumerate(gold_standards):
        if row[2] == "E1":
            e1_indices.append(krishnan_genes[:,0].tolist().index(row[0]))
    print(len(e1_indices))
    heatmap2 = torch.mean(heatmap2[e1_indices,:], dim = 0)
    
    heatmap3 = torch.load("mt_with_identity_newidset_moepli_randomlabelsExp" + name + "/idExpertProbabilities.pt")
    gold_standards = pd.read_csv("Data/RandomLabels/RandomPosID" + str(i) + ".csv").values
    krishnan_genes = pd.read_csv("Data/row-genes.txt").values
    e1_indices = []
    for index,row in enumerate(gold_standards):
        if row[2] == "E1":
            e1_indices.append(krishnan_genes[:,0].tolist().index(row[0]))
    print(len(e1_indices))
    heatmap3 = torch.mean(heatmap3[e1_indices,:], dim = 0)
    
    
    heatmap4 = torch.load("mt_with_identity_newidset_moepli_randomlabelsExp" + name + "/idExpertWeights.pt")
    gold_standards = pd.read_csv("Data/RandomLabels/RandomPosID" + str(i) + ".csv").values
    #krishnan_genes = pd.read_csv("Data/row-genes.txt").values
    e1_indices = []
    for index,row in enumerate(gold_standards):
        if row[2] == "E1":
            e1_indices.append(krishnan_genes[:,0].tolist().index(row[0]))
    print(len(e1_indices))
    heatmap4 = torch.mean(heatmap4[e1_indices,:], dim = 0)
    
    for i in range(52):
        outF.write(str(heatmap[i]) + ",")
    for i in range(52):
         outF.write(str(heatmap2[i]) + ",")
    for i in range(52):
         outF.write(str(heatmap3[i]) + ",")
    for i in range(52):
        if i != 51:
            outF.write(str(heatmap4[i]) + ",")
        else:
            outF.write(str(heatmap4[i]) + "\n")
outF.close()
            
