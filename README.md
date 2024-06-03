# DISGEN
Codes for DISGEN

## Dependencies

```bash

networkx=3.0
numpy=1.24.4
ogb=1.3.6
pandas=2.0.3
scikit-learn=1.3.0
scipy=1.10.1
torch=2.0.1+cu118
torch-cluster=1.6.1+pt20cu118
torch-geometric=2.3.1
torch-scatter=2.1.1+pt20cu118
torch-sparse=0.6.17+pt20cu118
torch-spline-conv=1.2.2+pt20cu118
torchvision=0.15.2+cu118
```



# Usage
## Clone the repository:

```bash
git clone https://github.com/GraphmindDartmouth/DISGEN.git
cd DISGEN/src
```


## Execute the following scripts to run data augmentation on different data:

```bash
python gnnexp.py --dataset {dataset} --dataset_root {dataset_root}  --train_val_test_idx_save_root {train_val_test_idx_save_root}
```
The data augmentation will generate augmented views of the original graph and save them in the dataset_root. 
{dataset_root} is also the path to save the dataset.

The train, validation, small and large test index will be saved in train_val_test_idx_save_root.

## Execute the following scripts to train DISGEN on bbbp:

```bash
python disentgnn.py --dataset {dataset} --dataset_root {dataset_root}  --train_val_test_idx_save_root {train_val_test_idx_save_root} --criterion {criterion}
```

## Execute the following scripts to train on other datas (PROTEINS, GraphSST2 and NCI1):
```bash
python disentgnn_proteins.py --dataset {dataset} --dataset_root {dataset_root}  --train_val_test_idx_save_root {train_val_test_idx_save_root} --criterion {criterion}
```
Here {dataset} can be one of the following: PROTEINS, GraphSST2 and NCI1. {criterion} is the loss function used to 
minimize the shared information between the hidden representation. The one used in paper is pair_loss_cos. It can also 
be selected from one of the following: 'pair_loss_triplet', 'pair_loss', 'cos_loss', 'pearson_loss', 'none'. 'none' means
no shared information loss is used. 

{dataset_root} is the root to save the dataset.

{train_val_test_idx_save_root} is the root to save the train, validation, small and large test index for the dataset.
