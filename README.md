# DISGEN
Codes for Enhancing Size Generalization in Graph Neural Networks through Disentangled Representation Learning (DISGEN)


# Introduction
We propose a general Disentangled representation learning framework for size Generalization (DISGEN) of GNNs. First, we
introduce new augmentation strategies to guide the model in learning relative size information. Second, we propose a 
decoupling loss to minimize the shared information between the hidden representations optimized for size- and 
task-related information, respectively.

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
Experiments are carried out on a NVIDIA L40 with CUDA Version 12.2

# Usage
## Clone the repository:

```bash
git clone https://github.com/GraphmindDartmouth/DISGEN.git
```
## Create an environment using conda:

```bash

conda create -n disgen python=3.8
conda activate disgen
pip install -r requirements.txt
```


## Execute the following scripts to use a gnn explainer and generate augmented graphs:

```bash
cd DISGEN/src
python gnnexp.py --dataset_root DISGEN/dataset  --train_val_test_idx_save_root DISGEN/saved_idx  --dataset bbbp
```
The data augmentation will generate augmented views of the original graph and save them in the {dataset} folder. 
{dataset} is also the path to save the dataset. {train_val_test_idx_save_root} is the path to save the train, validation, 
small and large test index for the dataset, and it will be used in the model training. {saved_model} is the path to save the pre-trained gnn model.

## Execute the following scripts to train DISGEN:

```bash
cd DISGEN/src
python disentgnn.py --dataset_root DISGEN/dataset  --train_val_test_idx_save_root DISGEN/saved_idx --saved_model DISGEN/saved_model --dataset bbbp --criterion pair_loss_cos
```

Here {criterion} is the loss function used to learn the relative size information, inspired by contrastive learning. It can also 
be selected from one of the following: 'pair_loss_triplet', 'pair_loss', 'cos_loss', 'pearson_loss', 'none'. 'none' means no relative size information loss is learned. 

You can also use run.sh to run the above codes
```
bash run.sh
```

## Contact
If you have any questions, suggestions, or bug reports, please contact
```
zheng.huang.gr@dartmouth.edu
```

