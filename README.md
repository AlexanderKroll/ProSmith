# ProSmith
This repository contains the code and datasets to reproduce the results and to train the ProSmith models of our manuscript "A multimodal transformer network for protein-small molecule interactions enhances predictions of drug-target affinities and enzyme substrates".



## Downloading data folder
Before you can reproduce the results of the manuscript, you need to [download and unzip a data folder from Zenodo](https://doi.org/10.5281/zenodo.10986299).
Afterwards, this repository should have the following strcuture:

    ├── code
    ├── data   
    ├── LICENSE.md     
    └── README.md

## Install
```
conda env create -f environment.yml
conda activate prosmith
pip install -r requirements.txt
```

## How to train and evaluate a model
### (a) Data preparation
First, you need a training, validation, and test set as csv files. Every csv file should be comma-separated and have the following columns:
-Protein sequence: contains the protein amino acid sequences of proteins
-SMILES: contains SMILES strings for small molecules
-output: The value of the target variable for the protein-small molecule pair of this row

For an example of such csv-files, please have a look at the files in following folder of this repositiory: "data/training_data/ESP/train_val".

### (b) Calculating input representations for proteins and small molecules:
Before you can start training ProSmith, ESM-1b embeddings and ChemBERTa2 embeddings need to be calculated for all protein sequences and SMILES strings in your repository, respectively. For example, for the ESP dataset this can be done by executing the following command:

```python
python /path_to_repository/code/preprocessing/preprocessing.py --train_val_path /path_to_repository/data/training_data/ESP/train_val/ \
															   --outpath /path_to_repository/data/training_data/ESP/embeddings \
															   --smiles_emb_no 2000 --prot_emb_no 2000
```
-"train_val_path": specifies, where all training and validation files are stored (with all protein sequences and SMILES strings)
-"outpath": specifies, where the calculated ESM-1b and ChemBERTa2 embeddings will be stored
-"smiles_emb_no" & "prot_emb_no": specify how many ESM-1b and ChemBERTa2 embeddings are stored in 1 file. 2000 for each variable worked well on a PC with 16GB RAM. Increased numbers might lead to better performance during model training.


### (c) Training the ProSmith Transformer Network:
To train the ProSmith Transformer Network (code is an example to train for the ESP task):

```python
python /path_to_repository/code/training/training.py --train_dir /path_to_repository/data/training_data/ESP/train_val/ESP_train_df.csv \
							    --val_dir /path_to_repository/data/training_data/ESP/train_val/ESP_train_df.csv \
							    --save_model_path /path_to_repository/data/training_data/ESP/saved_model \
							    --embed_path /path_to_repository/data/training_data/ESP/embeddings \
							    --pretrained_model /path_to_repository/data/training_data/BindingDB/saved_model/pretraining_IC50_6gpus_bs144_1.5e-05_layers6.txt.pkl \
							    --learning_rate 1e-5  --num_hidden_layers 6 --batch_size 24 --binary_task True \
							    --log_name ESP --num_train_epochs 100
```

This model will train for num_train_epochs=100 epochs and it will store the best model (i.e. with the best performance on the validation set) in "save_model_path". Therefore, after each epoch model performance is evaluated. 
"binary_task" is set to True, because the ESP prediction task is a binary classification task. The variable has to be set to False for regression tasks.


### (d) Training the Gradient Boosting Models:
To train gradient boosting models for the ESP task, execute the following command:

```python
python /path_to_repository/code/training/training_GB.py --train_dir /path_to_repository/data/training_data/ESP/train_val/ESP_train_df.csv \
							    --val_dir /path_to_repository/data/training_data/ESP/train_val/ESP_val_df.csv \
							    --test_dir /path_to_repository/data/training_data/ESP/train_val/ESP_test_df.csv \
							    --pretrained_model /path_to_repository/data/training_data/ESP/saved_model/ESP_1gpus_bs24_1e-05_layers6.txt.pkl \
							    --embed_path /path_to_repository/data/training_data/ESP/embeddings \
							    --save_pred_path /path_to_repository/data/training_data/ESP/saved_predictions \
							    --num_hidden_layers 6 --num_iter 500 --log_name ESP --binary_task True		    
```

The final predictions of the ProSmith model will be saved in "save_pred_path". They might differ from the original order in the csv file, but there is an additional file, containing the original indices. You can map the predictions to the csv file using the following python code:


```python
import numpy as np
import pandas as pd
from os.path import join

#loading predictions and test set:
y_pred = np.load(join(path_to_repository, "data","training_data", "ESP", "saved_predictions", "y_test_pred.npy"))
y_pred_ind = np.load(join(path_to_repository, "data","training_data", "ESP", "saved_predictions", "test_indices.npy"))
test_df = pd.read_csv(join(path_to_repository, "data, "training_data", ESP", "train_val", "ESP_test_df.csv"))

#Mapping predictions to test set:
test_df["y_pred"] = np.nan
for k, ind in enumerate(y_pred_ind):
    test_df["y_pred"][ind] = y_pred[k]
```


## Requirements for running the code in this GitHub repository
The code was implemented and tested on Linux with the following packages and versions
- python 3.8.3
- pandas 1.3.0
- torch 2.0.0+cu117
- numpy 1.22.4
- Bio 1.79
- transformers 4.27.2
- logging 0.5.1.2
- sklearn 1.2.2
- lifelines 0.27.7
- xgboost 0.90
- hyperopt 0.2.5
- json 2.0.9



 
