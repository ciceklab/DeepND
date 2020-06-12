# DeepND
DeepND ASD &amp; ID 

## System Requirements
### Hardware Requirements

<b>GPU:</b><br/>
- For all spatio temporal regions : 7 nVidia GeForce GTX 1080 or equivalent configurations with at least 70GB of memory;  Recommended: 3 nVidia Titan RTx<br/>
- For the Example included in this repository : 1 nVidia GeForce GTX 1080 or equivalent configurations<br/>

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

## Install

In order to install required packages, either one of following methods can be used. <br/>
<b>Anaconda Quick Installation:</b>  After downloading all files, import the conda enviroment named "deepnd-legacy.yml" and activate.<br/>
<b>General Installation:</b> Install all packages listed under "Software Requirements : Dependencies" section with their given versions. 

## Usage

The model parameters are listed in "main.py". 
- root : Working directory to read data and store results | Default : ""<br/>
- trial : Number of trials to train | Default : 10<br/>
- k : k-fold cross validation | Default : 5<br/>
- mode : Mode selection. In the train mode randomly initialized model is trained. In the test mode, previously saved models are used to regenerate results. | 1 : Test mode, 0: Train mode | Default : 0<br/>
- experiment : The experiment ID. Significant for test mode as the folder with the same id will be used to load random states and models. | Default : 0<br/>
- model_select : Single Task or Multi Task model selection. For single task model the disease should eb specified as either ID or ASD by setting the proper parameter. | 1 : Multi, 0: Single | Default : 1<br/>
- disease : The name of the subject disease. It is required for single task model (DeepND-ST), can be left in default for Multi Task model (DeepND)  | 0 : ASD, 1 : ID | Default : 0<br/>
- networks : List that contains spatio-temporal regions to be fed to the model. The example is given for region 11 (temporal window 12-14), to run the example : [11] | Default (all regions) : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] |<br/>

After setting the parameters listed above, you can run the program using the following command in working directory:
```
python main.py
```
