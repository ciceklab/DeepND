# DeepND 
[![DOI](https://zenodo.org/badge/271965480.svg)](https://zenodo.org/badge/latestdoi/271965480)

## Contents
- [Overview](https://github.com/ciceklab/DeepND#overview)
    - [Authors](https://github.com/ciceklab/DeepND#authors)
    - [Questions & Comments](https://github.com/ciceklab/DeepND#questions--comments)
- [Repository Contents](https://github.com/ciceklab/DeepND#repository-contents)
- [System Requirements](https://github.com/ciceklab/DeepND#system-requirments)
- [Installation Guide](https://github.com/ciceklab/DeepND#installation-guide)
- [Usage](https://github.com/ciceklab/DeepND#usage)
- [Demo](https://github.com/ciceklab/DeepND#demo)
- [Replicating Published Results](https://github.com/ciceklab/DeepND#replicating-published-results)
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

DeepND can run on either CPU or GPU. However, for shorter training time, we strongly recommend GPU support of at least one Nvidia GeForce GTX 1080 or its equivalent.

### Software Requirments
<b>Python:</b> 3.6+<br/>
<b>OS:</b> All major platforms (Linux, macOS, Windows)<br/>
<b>Dependencies:</b> Listed in ```requirements.txt``` <br/>

## Installation Guide

In order to install required packages, either one of following methods can be used. <br/>
<b>Anaconda Quick Installation:</b>  After downloading all files, import the conda enviroment named "deepnd_legacy_environment.yml" and activate. Typical install time is 10 minutes. <br/>

```
$ conda env create -f deepnd_legacy_environment.yml
$ conda activate deepnd-legacy
```

<b>General Installation:</b> Install all packages listed under [Software Requirements-Dependencies](https://github.com/ciceklab/DeepND#software-requirments) section with their given versions with either PyPI or Anaconda.

```
$ pip install -r requirements.txt
```
OR

```
$ conda install -r requirements.txt
```

## Usage

The model parameters are explained in the configuration file.
You can run the program using the following command in working directory:
```
python main.py
```

## Demo

A quick demo for DeepND is included for autism spectrum disorder (ASD) and intelectual disabilty(ID). This demo is designed to generate predictions for ASD and ID on an average personal computer. Therefore, the inputs are limited to pLI and a single network which is the brain gene co-expression network for MDCBC region between adolescence and mid-adulthood. 

Required inputs, such as spatio-temporal brain co-expression networks, gene features and gold standards are provided in the "Data" folder.  All of the reqired parameters are already set in the <i>config</i> file as default values.

1. Download all files and installing required packages as explained under [Installation Guide](https://github.com/ciceklab/DeepND#installation-guide).

2. Run the following terminal command:

```
python main.py
```
This command will run the main function of the DeepND which sorts gold standard gene lists, creates randomized masks for the cross-validation, initializes the model and starts training.

After each training, the best model is saved as a ".pth" file to reproduce results later on and to enable more advanced trainin procedures (such as transfer learning) if needed. Before each trial, the allocated and cached GPU memory is reported so that you monitor and adjust your inputs according to your own compuatational units limitations.

An example terminal output of the main process would be as follows.

![Example terminal output](https://github.com/ciceklab/DeepND/blob/master/example.png)

3. The output will be written to MultiExp00 folder which contains:
    - Saved models for each training as ".pth" files
    - Prediction results for first disorder (ASD), and the second (ID) in separate .txt files
    - Experimental stats (mean, stdev and median of performance metrics, runtime etc.) in runreport.txt file
    - Tensors that highlight weights of spatio-temporal brain networks for different subset of genes as ".pt" files
    - pyTorch and Numpy random states for reproducing the same results in test mode

You may also try running this demo with different gene interactions networks. Several examples networks that are mentioned in the artice are publicly avaliable at  [https://doi.org/10.5281/zenodo.3892979](https://doi.org/10.5281/zenodo.3892979). You can download any number of networks from there, and adjust the <i>config</i> file accordingly before repeating Step 2.

## Replicating Published Results

This Github page is part of the work included in ["Deep multitask learning of gene risk for comorbid neurodevelopmental disorders"](https://www.biorxiv.org/content/10.1101/2020.06.13.150201v3). To replicate the results presented in the paper, please refer to [Replicating Results](https://github.com/ciceklab/DeepND/blob/master/replicate.md) .

## License
- CC BY-NC-SA 2.0
- To use the information on this repository, please cite the manuscript covering this work.
    ```
    @article{deepND2021,
    title={Deep multitask learning of gene risk for comorbid neurodevelopmental disorders},
    author={Beyreli, Ilayda and Karakahya, Oguzhan and Cicek, A Ercument},
    journal={bioRxiv},
    pages={2020--06},
    year={2021},
    publisher={Cold Spring Harbor Laboratory}
    }
    ```
- For commercial usage, please contact.
