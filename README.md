# Robust Optimization for Fairness with Noisy Protected Groups
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/wenshuoguo/robust-fairness-code/blob/master/LICENSE)
Code for experiments in paper [Robust Optimization for Fairness with Noisy Protected Groups](https://arxiv.org/pdf/2002.09343.pdf), NeurIPS 2020.
## Prerequisites
Python 3, tensorflow 1.14.0, numpy, pandas
## Data preprocessing
### Adult dataset
The Adult dataset is public and available [here](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/).
The function ```preprocess_data_adult``` in```data.py``` is used to preprocess the dataset. The preprocessed Adult dataset ```adult_processed.csv``` is included.
### Taiwan Credit dataset
The Taiwan Credit dataset is public and available [here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).
The function ```preprocess_data_credit``` in```data.py``` is used to preprocess the dataset. The function ```load_dataset_credit``` loads the dataset and binarizes the features. The preprocessed binarized Credit dataset ```credit_default_processed.csv``` is included.

## Running the experiments
We provide general procedure to load data, import supplementary ```.py``` files and set global variables. Then we give instructions on running each individual algorithm.

### Load data
Import: ```data.py, losses.py, optimization.py, model.py, utils.py, tensorflow, numpy```

Run: ```df = data.load_dataset_adult()``` or ```df = data.load_dataset_credit()```

### Set variables: 
```
LABEL_COLUMN = "label" (for Adult); "default" (for Credit)
FEATURE_NAMES = list(df.keys())
FEATURE_NAMES.remove(LABEL_COLUMN)
PROTECTED_COLUMNS = ['race_White', 'race_Black', 'race_Other_combined'] (for Adult); ['EDUCATION_grad', 'EDUCATION_uni', 'EDUCATION_hs_other'] (for Credit)
```
To set the variables for the protected groups and proxy groups:

For Oracle baseline algirithm without noise:
```
PROXY_COLUMNS = PROTECTED_COLUMNS 
```

For Adult/Credit dataset with a noise parameter:

```PROXY_COLUMNS = data.get_proxy_column_names(PROTECTED_COLUMNS, noise_parameter)```

### Naive algorithm

Additional import:```naive_training.py```

Run the algrithm:

```naive_training.get_results_for_learning_rates(df, FEATURE_NAMES, PROTECTED_COLUMNS, PROXY_COLUMNS, LABEL_COLUMN,constraint='tpr')```
with a list of learning rates. Please use ``` constraint='tpr_and_fpr'``` for experiments on Credit dataset.


### DRO algorithm

Additional import:```dro_training.py```

Run the algrithm:

```dro_training.get_results_for_learning_rates(df, FEATURE_NAMES, PROTECTED_COLUMNS, PROXY_COLUMNS, LABEL_COLUMN,constraint='tpr')```
with a list of learning rates.


### Softweights algorithm 

Additional import:```softweights_training.py```

Run the algrithm:

```softweights_training.get_results_for_learning_rates(df, FEATURE_NAMES, PROTECTED_COLUMNS, PROXY_COLUMNS, LABEL_COLUMN,constraint='tpr')```
with a list of learning rates.

