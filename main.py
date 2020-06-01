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

###############################################################################################################################################
"""LOADING NETWORKS"""
###############################################################################################################################################
network_names = ["PFC", "MDCBC","V1C", "SHA"]
periods = ["1-3", "2-4", "3-5", "4-6", "5-7", "6-8", "7-9", "8-10", "9-11", "10-12", "11-13", "12-14", "13-15"]
pfc08Mask = [i for i in range(network_count)]
mdcbc08Mask = [i for i in range(network_count)]
v1c08Mask = [i for i in range(network_count)]
sha08Mask= [i for i in range(network_count)]

for period in pfc08Mask:
    pfcnetworks.append(torch.load(root + "/Data/EdgeTensors/PointEight/PFC" + periods[period] + "wTensor.pt").type(torch.LongTensor))
    pfcnetworkweights.append(torch.abs(torch.load((root + "/Data/EdgeTensors/PointEight/PFC" + periods[period] + "EdgeWeightTensor.pt").type(torch.FloatTensor)[0,:]))

for period in mdcbc08Mask:
    mdcbcnetworks.append(torch.load(root + "/Data/EdgeTensors/PointEight/MDCBC" + periods[period] + "wTensor.pt").type(torch.LongTensor))
    mdcbcnetworkweights.append(torch.abs(torch.load(root + "/Data/EdgeTensors/PointEight/MDCBC" + periods[period] + "EdgeWeightTensor.pt").type(torch.FloatTensor)[0,:]))
  
for period in v1c08Mask:
    v1cnetworks.append(torch.load(root + "/Data/EdgeTensors/PointEight/V1C" + periods[period] + "wTensor.pt").type(torch.LongTensor)) 
    v1cnetworkweights.append(torch.abs(torch.load(root + "/Data/EdgeTensors/PointEight/V1C" + periods[period] + "EdgeWeightTensor.pt").type(torch.FloatTensor)[0,:]))

for period in sha08Mask:
    shanetworks.append(torch.load(root + "/Data/EdgeTensors/PointEight/SHA" + periods[period] + "wTensor.pt").type(torch.LongTensor)) 
    shanetworkweights.append(torch.abs(torch.load(root + "/Data/EdgeTensors/PointEight/SHA" + periods[period] + "EdgeWeightTensor.pt").type(torch.FloatTensor)[0,:]))

for i in range(network_count):
    pfcnetworks[i] = pfcnetworks[i].to(devices[pfcgpumask[i]])
    pfcnetworkweights[i] = pfcnetworkweights[i].to(devices[pfcgpumask[i]])
    mdcbcnetworks[i] = mdcbcnetworks[i].to(devices[mdcbcgpumask[i]])
    mdcbcnetworkweights[i] = mdcbcnetworkweights[i].to(devices[mdcbcgpumask[i]])
    v1cnetworks[i] = v1cnetworks[i].to(devices[v1cgpumask[i]])
    v1cnetworkweights[i] = v1cnetworkweights[i].to(devices[v1cgpumask[i]])
    shanetworks[i] = shanetworks[i].to(devices[shagpumask[i]])
    shanetworkweights[i] = shanetworkweights[i].to(devices[shagpumask[i]])
                             
geneNames_all = pd.read_csv(root + "/Data/Brainspan/row-genes.txt", header = None)
geneNames_all = geneNames_all[0].tolist()
geneDict = constructGeneDictionary(root + "/Data/Brainspan/hugogenes_entrez.txt")
