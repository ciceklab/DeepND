# DeepND 
## Contents
- [Overview](https://github.com/ciceklab/DeepND#overview)
- [Repository Contents](https://github.com/ciceklab/DeepND#repository-contents)
- [System Requirements](https://github.com/ciceklab/DeepND#system-requirments)
- [Installation Guide](https://github.com/ciceklab/DeepND#installation-guide)
- [Usage](https://github.com/ciceklab/DeepND#usage)
- [Demo](https://github.com/ciceklab/DeepND#demo)
- [License](https://github.com/ciceklab/DeepND#license)

## Overview

DeepND is a cross-disorder gene discovery algorithm (Deep Neurodevelopmenal Disorders algorithm) It analyzes comorbid neurodevelopmental disorders simultaneously and explicitly learns the shared and disorder-specific genetic features using multitask learning. Thus, the predictions for the disorders depend on each other's genetic architecture. The proposed DeepND architecture uses graph convolution to extract associations between genes from gene coexpression networks that model neurodevelopment. This information is processed by a mixture-of-experts model that can self-learn critical neurodevelopmental time windows and brain regions for each disorder etiology which makes the model interpretable. We provide a genomewide risk ranking for each disorder.

The model can easily be extended to work with more than 2 comorbid disorders or with other types of gene interaction networks. The model can also work in a singletask mode to analyze disorders independently which is referred as DeepND-ST.

---

## Authors

Ilayda Beyreli*, Oguzhan karakahya*, A. Ercument Cicek

## Questions & comments 

[firstauthorname].[firstauthorsurname]@bilkent.edu.tr
[secondauthorname].[secondauthorsurname]@bilkent.edu.tr

## Repository Contents
- Data : Feature sets, gold standards and example networks
- ASDExp00 : Example output for DeepND-ST
- MultiExp00 : Example output for DeepND
## System Requirements
### Hardware Requirements

<b>GPU:</b><br/>
- For all spatio temporal regions : 7 NVIDIA GeForce GTX 1080 Ti or equivalent configurations with at least 70GB of memory;  Recommended: 3 NVIDIA TITAN RTX<br/>
- For the demo included in this repository : 1 NVIDIA GeForce GTX 1080 Ti or equivalent configurations<br/>

<b>Disk Storage:</b> 25 GB of free disk space

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

The model parameters are listed in "main.py". 
- root : Working directory to read data and store results | Default : ""<br/>
- trial : Number of trials to repeat training with random weight initializations | Default : 10<br/>
- k : k-fold cross validation | Default : 5<br/>
- mode : Mode selection. In the train mode randomly initialized model is trained. In the test mode, previously saved models are used to regenerate results. | 1 : Test mode, 0: Train mode | Default : 0<br/>
- experiment : The experiment ID. Significant for test mode as the folder with the same id will be used to load random states and models. | Default : 0<br/>
- model_select : Single Task or Multi Task model selection. For single task model the disease should be specified as either ID or ASD by setting the proper parameter. | 1 : Multi, 0: Single | Default : 1<br/>
- disorder : The name of the subject disorder. It is required for singletask mode (DeepND-ST), can be left in default for multitask mode (DeepND)  | 0 : ASD, 1 : ID | Default : 0<br/>
- networks : List that contains spatio-temporal regions to be fed to the model. The example is given for region 11 (temporal window 12-14), to run the example : [11] | Default (all regions) : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] |<br/>

After setting the parameters listed above, you can run the program using the following command in working directory:
```
python main.py
```

## Demo

A quick demo for DeepND is included for autism spectrum disorder (ASD) and intelectual disabilty(ID). Required inputs, such as spatio-temporal brain co-expression netwroks, gene features and gold standards are provided in the "Data" folder. 

### DeepND Train Mode

1. After downloading all files and installing required packages as explained under [Installation Guide](https://github.com/ciceklab/DeepND#installation-guide), use the configuration below for "main.py". 

```
root = ""
trial = 10
k = 5
mode = 0
experiment = 0
model_select = 1
disorder = 0
networks = [11] 
pfcgpumask = [0]
mdcbcgpumask = [0]
v1cgpumask = [0]
shagpumask = [0]
```

2. Run the following terminal command:
```
python main.py
```

3. The output will be written to MultiExp00 folder which contains:
    - Saved models for each training as ".pth" files
    - Prediction results for first disorder (ASD), and the second (ID) in separate .txt files
    - Experimental stats (mean, stdev and median of performance metrics, runtime etc.) in runreport.txt file
    - heatmaps that highlight weights of spatio-temporal brain netwroks for different subset of genes as ".pt" files
    - pyTorch and Numpy random states for reproducing the same results in test mode
  
### DeepND Test Mode

1. After downloading all files and installing required packages as explained under [Installation Guide](https://github.com/ciceklab/DeepND#installation-guide), use the configuration below for "main.py". 

```
root = ""
trial = 10
k = 5
mode = 1
experiment = 0
model_select = 1
disorder = 0
networks = [11] 
pfcgpumask = [0]
mdcbcgpumask = [0]
v1cgpumask = [0]
shagpumask = [0]
```

2. Run the following terminal command:
```
python main.py
```

3. The program will read the saved models and random states from MultiExp00 and the output will be written to MultiExp00Test folder which contains:
    - Prediction results for first disorder (ASD), and the second (ID) in separate .txt files
    - Experimental stats (mean, stdev and median of performance metrics, runtime etc.) in runreport.txt file
    - heatmaps that highlight weights of spatio-temporal brain netwroks for different subset of genes as ".pt" files
  
### DeepND-ST Train Mode

1. After downloading all files and installing required packages as explained under [Installation Guide](https://github.com/ciceklab/DeepND#installation-guide), use the configuration below for "main.py". 

```
root = ""
trial = 10
k = 5
mode = 0
experiment = 0
model_select = 0
disorder = 0
networks = [11] 
pfcgpumask = [0]
mdcbcgpumask = [0]
v1cgpumask = [0]
shagpumask = [0]
```

2. Run the following terminal command:
```
python main.py
```

3. The output will be written to ASDExp00 folder which contains:
    - Saved models for each training as ".pth" files
    - Prediction results for the specified disorder as a .txt file
    - Experimental stats (mean, stdev and median of performance metrics, runtime etc.) in runreport.txt file
    - heatmaps that highlight weights of spatio-temporal brain netwroks for different subset of genes as ".pt" files
    - pyTorch and Numpy random states for reproducing the same results in test mode
  
### DeepND-ST Test Mode

1. After downloading all files and installing required packages as explained under [Installation Guide](https://github.com/ciceklab/DeepND#installation-guide), use the configuration below for "main.py". 

```
root = ""
trial = 10
k = 5
mode = 1
experiment = 0
model_select = 0
disorder = 0
networks = [11] 
pfcgpumask = [0]
mdcbcgpumask = [0]
v1cgpumask = [0]
shagpumask = [0]
```

2. Run the following terminal command:
```
python main.py
```

3. The program will read the saved models and random states from ASDExp00 and the output will be written to ASDExp00Test folder which contains:
    - Prediction results for the specified disorder (ASD) as a .txt file
    - Experimental stats (mean, stdev and median of performance metrics, runtime etc.) in runreport.txt file
    - heatmaps that highlight weights of spatio-temporal brain netwroks for different subset of genes as ".pt" files

### Reproducing the results given in the manuscript

**Warning:** Make sure your computational unit satifies the hardware requirements for full experiment which is listed under [Requirements](https://github.com/ciceklab/DeepND#system-requirements).

1. After downloading all files and installing required packages as explained under [Installation Guide](https://github.com/ciceklab/DeepND#installation-guide), use the configuration below for "main.py". Note that `pfcgpumask`, `mdcbcgpumask`, `v1cgpumask` and `shagpumask` are GPU mappings for temporal networks of those regions and they depend on GPU specifications of the computational unit i.e. number of GPUs, avaliable memory of each GPU etc. The setup given below is for 7 NVIDIA GeForce GTX 1080 Ti and you may need to adjust it manually for your own setup.

```
root = ""
trial = 10
k = 5
mode = 0
experiment = 1
model_select = 0
disorder = 0
networks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
pfcgpumask = [0,0,0,0,5,5,5,6,6,7,7,7,7]
mdcbcgpumask = [0,1,2,2,2,3,3,3,3,4,4,4,4]
v1cgpumask = [4,4,4,4,4,5,5,5,5,5,5,5,5]
shagpumask = [6,6,6,6,6,6,7,7,7,7,7,7,7]
```

2. Run the following terminal command:
```
python main.py
```

3. The output will be written to MultiExp01 folder which contains:
    - Saved models for each training as ".pth" files
    - Prediction results for first disorder (ASD), and the second (ID) in separate .txt files
    - Experimental stats (mean, stdev and median of performance metrics, runtime etc.) in runreport.txt file
    - heatmaps that highlight weights of spatio-temporal brain netwroks for different subset of genes as ".pt" files
    - pyTorch and Numpy random states for reproducing the same results in test mode 

## License
- CC BY-NC-SA 2.0
- For commercial usage, please contact.
