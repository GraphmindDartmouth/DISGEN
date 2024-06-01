# DISGEN
Codes for DISGEN

## Dependencies

```bash

networkx=3.0
numpy=1.24.4
ogb=1.3.6
pandas=2.0.3
pillow=9.3.0
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


## Execute the following scripts to run data augmentation on bbbp data:

```bash
python gnnexp.py
```

## Execute the following scripts to train on bbbp data:

```bash
python disentgnn.py
```

Execute the following scripts to train on other datas (PROTEINS, GraphSST2 and NCI1):
```bash
python disentgnn_proteins.py --dataset_name PROTEINS
```

