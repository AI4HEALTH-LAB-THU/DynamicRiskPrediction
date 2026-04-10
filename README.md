# From snapshots to trajectories in chronic disease risk stratification
A clinically interpretable and model-agnostic framework for chronic disease risk stratification using longitudinal physiological features

01_ModelDevelopmentValidation: The Python code in this directory pertains to the workflow of model development and validation.

02_StatisticalAnalysis: The R code in this directory is for the procedures of statistical analysis and visualization.

This repository contains the source code for our study on dynamic risk prediction. The codebase includes data preprocessing, feature engineering, feature selection, hyperparameter optimization, model training, and model evaluation workflows.

## 1. System requirements

### 1.1 Software dependencies and operating systems

The software has been developed and tested in Python. The main dependencies are listed below:

- Python 3.10
- numpy 1.26.4
- pandas 2.2.2
- scipy 1.13.1
- scikit-learn 1.5.2
- xgboost 2.1.3
- optuna 3.6.1
- joblib 1.4.2
- Boruta 0.4.3

Supported operating systems:
- Ubuntu 21.0

### 1.2 Versions the software has been tested on

The software has been tested on the following environments:

- Python 3.10.13 on Ubuntu 21.0

### 1.3 Required non-standard hardware

No non-standard hardware is required for running the demo.

For full-scale training on larger datasets, a workstation with at least:
- 16 GB RAM
- multi-core CPU

is recommended.

## 2. Installation guide

### 2.1 Instructions

#### Step 1. Clone the repository


git clone https://github.com/AI4HEALTH-LAB-THU/DynamicRiskPrediction.git
cd DynamicRiskPrediction

#### Step 2. Create a virtual environment (recommended)
Using venv:

python -m venv venv
source venv/bin/activate


## 3. Demo


A small demo dataset is provided to illustrate the workflow and allow users and reviewers to test the software without access to the full study dataset.

### 3.1 Instructions to run on data

Please place the demo data in the designated folder, for example:

data/train_test_dataset/bench_dynamic_251206/MI_Sample_Data/feather

### 3.2 Expected output

After successful execution, the software will generate output files in the output directory. These include:
	•	predicted risk scores for each sample
	•	model evaluation metrics
	•	serialized trained model objects
	•	logs of training or inference

Typical evaluation outputs include:
	•	AUROC
	•	AUPRC
	•	Brier score
	•	selected features
	•	optimized hyperparameters

### 3.3 Expected run time for demo on a “normal” desktop computer

The demo typically completes within:
	•	40–80 minutes on a 32 CPUs server

The exact run time depends on CPU performance and the size of the demo dataset.

## 4. Instructions for use

### 4.1 How to run the software on your data

To apply the software to your own data, prepare an input file in feather format with the same variable names, coding scheme, and structure expected by the scripts.

General workflow
	1.	Prepare the input dataset
	2.	Perform preprocessing
	3.	Run feature selection if required
	4.	Train the model or load a pretrained model
	5.	Generate predictions
	6.	Evaluate model performance

![Fig1](https://github.com/user-attachments/assets/9cbbee30-b346-4420-bce3-56d52a1bd3f2)

