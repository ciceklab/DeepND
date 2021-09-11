# DeepND 
## Contents
- [Overview](https://github.com/ciceklab/DeepND#overview)
    - [Authors](https://github.com/ciceklab/DeepND#authors)
    - [Questions & Comments](https://github.com/ciceklab/DeepND#questions--comments)
- [Repository Contents](https://github.com/ciceklab/DeepND#repository-contents)
- [System Requirements](https://github.com/ciceklab/DeepND#system-requirments)
- [Installation Guide](https://github.com/ciceklab/DeepND#installation-guide)
- [Usage](https://github.com/ciceklab/DeepND#usage)
- [Demo](https://github.com/ciceklab/DeepND#demo)
    - [Reproducing the results in the manuscript](https://github.com/ciceklab/DeepND#reproducing-the-results-given-in-the-manuscript)
- [License](https://github.com/ciceklab/DeepND#license)

## Overview

DeepND is a cross-disorder gene discovery algorithm (Deep Neurodevelopmenal Disorders algorithm) It analyzes comorbid neurodevelopmental disorders simultaneously and explicitly learns the shared and disorder-specific genetic features using multitask learning. Thus, the predictions for the disorders depend on each other's genetic architecture. The proposed DeepND architecture uses graph convolution to extract associations between genes from gene coexpression networks that model neurodevelopment. This information is processed by a mixture-of-experts model that can self-learn critical neurodevelopmental time windows and brain regions for each disorder etiology which makes the model interpretable. We provide a genomewide risk ranking for each disorder.

The model can easily be extended to work with more than 2 comorbid disorders of any kind (not necessarily neurodevelopmental disorders) or with other types of gene interaction networks. The model can also work in a singletask mode to analyze disorders independently which is referred as DeepND-ST.

---

## Authors

Ilayda Beyreli*, Oguzhan karakahya*, A. Ercument Cicek

## Questions & Comments 

[firstauthorname].[firstauthorlastname]@bilkent.edu.tr or <br>
[secondauthorfirstinitial].[secondauthorlastname]@bilkent.edu.tr

## Repository Contents
- Data : Feature sets, gold standards and example networks
- ASDExp00 : Example output for DeepND-ST when analyzing genomewide Autism Spectrum Disorder (ASD) risk.
- MultiExp00 : Example output for DeepND when co-analyzing genomewide ASD and Intellectual Disability (ID) risk.
## System Requirements
### Hardware Requirements

DeepND can run on either CPU or GPU. However, for shorter training time, we strongly recommend GPU support.

### Software Requirments
<b>Python:</b> 3.6+<br/>
<b>OS:</b> All major platforms (Linux, macOS, Windows)<br/>
<b>Dependencies:</b><br/>
cudatoolkit               10.0.130<br/>
pytorch                   1.1.0<br/>
torch-cluster             1.4.2<br/>
torch-geometric           1.2.1<br/>
torch-scatter             1.2.0<br/>
torch-sparse              0.4.0<br/>
torch-spline-conv         1.1.0<br/>
torchvision               0.3.0<br/>
networkx                  2.4<br/>
numpy<br/>
scipy<br/>
pandas<br/>
seaborn<br/>
scikit-learn<br/>
pillow<br/>

## Installation Guide

In order to install required packages, either one of following methods can be used. <br/>
<b>Anaconda Quick Installation:</b>  After downloading all files, import the conda enviroment named "deepnd-legacy.yml" and activate. Typical install time is 10 minutes. <br/>

```
$ conda env create -f deepnd-legacy.yml
$ conda activate deepnd-legacy
```

<b>General Installation:</b> Install all packages listed under [Software Requirements-Dependencies](https://github.com/ciceklab/DeepND#software-requirments) section with their given versions. 

## Usage

The model parameters are explained in the configuration file.
You can run the program using the following command in working directory:
```
python main.py
```

## Demo

A quick demo for DeepND is included for autism spectrum disorder (ASD) and intelectual disabilty(ID). Required inputs, such as spatio-temporal brain co-expression netwroks, gene features and gold standards are provided in the "Data" folder. 

1. After downloading all files and installing required packages as explained under [Installation Guide](https://github.com/ciceklab/DeepND#installation-guide), run the following terminal command:

```
python main.py
```

2. The output will be written to MultiExp00 folder which contains:
    - Saved models for each training as ".pth" files
    - Prediction results for first disorder (ASD), and the second (ID) in separate .txt files
    - Experimental stats (mean, stdev and median of performance metrics, runtime etc.) in runreport.txt file
    - Tensors that highlight weights of spatio-temporal brain netwroks for different subset of genes as ".pt" files
    - pyTorch and Numpy random states for reproducing the same results in test mode

## Replicating Published Results

[Replicating Results](https://github.com/ciceklab/DeepND/blob/master/replicate.md)

## License
- CC BY-NC-SA 2.0
- For commercial usage, please contact.
