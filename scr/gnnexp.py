import argparse
import torch
import os
import datapreprocessing
import numpy as np
import random
import pickle
# import wandb

import util_new
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import MoleculeNet, TUDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import MoleculeNet
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.explain.metric import fidelity
from torch_geometric.utils import to_undirected

from graphsst2_dataset import get_dataset
from torch_geometric.data import DataLoader, Batch
from model import StructuralGCN, BaselineGCN, MultiViewGCN
from train_util import train_step, eval_step, train_data


torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="bbbp", help='datasets to consider')
parser.add_argument('--patience', default=50, type=int, help='patience for early stopping')
parser.add_argument('--model', default="BaselineGCN", help='models to consider')
parser.add_argument('--dataset_root', default="/dataset",
                    help='datasets root')

parser.add_argument('--train_val_test_idx_save_root',
                    default="/saved_idx",
                    help='root to save train val test idx')


parser.add_argument('--batch_size', default=32, type=int, help='patience for early stopping')
parser.add_argument('--gpu_id', default="0",  help='GPU id')
parser.add_argument('--saved_model', default="/saved_model",  help='path to save pre-trained gnn')
parser.add_argument('--lr', default=0.001,  help='learning rate')


# config_wandb = {}
# wandb.init(project="project_name", config=config_wandb)
# wandb.init(mode="disabled")

args = parser.parse_args()

dataset_root = args.dataset_root
train_val_test_idx_save_address = args.train_val_test_idx_save_root

batch_size = args.batch_size
# name: "hiv", "bbbp", "graphsst2"
dataset = args.dataset
structural_feature_name = args.structural_feature_name

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print("Working on data: ", dataset)
if dataset == "bbbp":
    cur_data = PygGraphPropPredDataset(name="ogbg-molbbbp", pre_transform=None, root=dataset_root)
    dataset_root = dataset_root + "_bbbp"

elif dataset == "nci1":
    cur_data = TUDataset(root=dataset_root, name='NCI1', use_node_attr=True)
    dataset_root = dataset_root + "_nci1"

elif dataset == "PROTEINS":
    cur_data = TUDataset(root=dataset_root, name='PROTEINS', use_node_attr=True)
    dataset_root = dataset_root + "_PROTEINS"

elif dataset == "graphsst2":
    cur_data = get_dataset(dataset_dir=dataset_root, dataset='Graph-SST2')  # 70,042 , 2 label, feature: yes
    dataset_root = dataset_root + "_graphsst2"

else:
    raise NotImplementedError("Dataset name error")


print("You are using: ", args.structural_feature_name)


train_val_test_idx_save_address = os.path.join(train_val_test_idx_save_address, dataset)

if not os.path.exists(train_val_test_idx_save_address):
    os.makedirs(train_val_test_idx_save_address)
    train_idx, val_idx, small_test_idx, large_test_idx = datapreprocessing.quick_test_data_prep(cur_data)
    datapreprocessing.save_idx_files(train_val_test_idx_save_address, train_idx, val_idx,
                                                small_test_idx, large_test_idx)
    with open(f"{train_val_test_idx_save_address}/train_idx.pkl", "rb") as file:
        silo_train_idx = pickle.load(file)
    with open(f"{train_val_test_idx_save_address}/val_idx.pkl", "rb") as file:
        silo_val_idx = pickle.load(file)

    with open(f"{train_val_test_idx_save_address}/small_test_idx.pkl", "rb") as file:
        small_test_idx = pickle.load(file)

    with open(f"{train_val_test_idx_save_address}/large_test_idx.pkl", "rb") as file:
        large_test_idx = pickle.load(file)

else:
    with open(f"{train_val_test_idx_save_address}/train_idx.pkl", "rb") as file:
        train_idx = pickle.load(file)
    with open(f"{train_val_test_idx_save_address}/val_idx.pkl", "rb") as file:
        val_idx = pickle.load(file)

    with open(f"{train_val_test_idx_save_address}/small_test_idx.pkl", "rb") as file:
        small_test_idx = pickle.load(file)

    with open(f"{train_val_test_idx_save_address}/large_test_idx.pkl", "rb") as file:
        large_test_idx = pickle.load(file)


print(len(train_idx), len(val_idx), len(small_test_idx), len(large_test_idx))
train_dataset = cur_data[train_idx]
util_new.draw_stat(train_dataset)

val_dataset = cur_data[val_idx]
small_test_dataset = cur_data[small_test_idx]
util_new.draw_stat(small_test_dataset)

large_test_dataset = cur_data[large_test_idx]
util_new.draw_stat(large_test_dataset)

x_s = train_dataset[0].x.shape[-1]
y_s = train_dataset[0].y.shape[-1]

model_name = args.model
if model_name == "BaselineGCN":
    model = BaselineGCN(x_s, y_s)
else:
    raise NotImplementedError("Model not implemented!")

cls_criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

model_save_path = "/saved_model"
if os.path.isfile(model_save_path + model_name + "_saved.pt"):
    model = torch.load(model_save_path + model_name + "_saved.pt")
    print("Model loaded!!")
else:
    print("No model saved, start training!!")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader_small = DataLoader(small_test_dataset, batch_size=batch_size, shuffle=False)
    test_loader_large = DataLoader(large_test_dataset, batch_size=batch_size, shuffle=False)

    train_data(train_loader, val_loader, test_loader_small, test_loader_large, model, optimizer, device)
    torch.save(model, model_save_path + model_name + "_saved.pt")
    print("Model saved!!")


explainer_model = model
train_dataset_ori = datapreprocessing.AugmentedDatasetOri(train_dataset, dataset_root, "train", device)
train_dataset_label_change = datapreprocessing.AugmentedDatasetLabelChange(train_dataset, dataset_root,
                                                                                      "train", explainer_model,
                                                                                      cls_criterion,
                                                                                      device)

train_dataset_label_unchange = datapreprocessing.AugmentedDatasetLabelUnChange(train_dataset, dataset_root,
                                                                                          "train", explainer_model,
                                                                                          cls_criterion,
                                                                                          device)


data_list = []
for i in range(len(train_dataset_ori)):
    # print(train_dataset_label_change[i].x.shape)
    # print(train_dataset_label_change[i].edge_index.shape)
    # print(train_dataset_label_unchange[i].x.shape)
    # print(train_dataset_label_unchange[i].edge_index.shape)
    data_list.append(datapreprocessing.PairData(x=train_dataset_ori[i].x, edge_index=train_dataset_ori[i].edge_index,
                                          y=train_dataset_ori[i].y,
                                          x_label_change=train_dataset_label_change[i].x,
                                          edge_index_label_change=train_dataset_label_change[i].edge_index,
                                          x_label_unchange=train_dataset_label_unchange[i].x,
                                          edge_index_label_unchange=train_dataset_label_unchange[i].edge_index))

train_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, follow_batch=['x_ori', 'x_label_change', 'x_label_unchange'])
batch = next(iter(train_loader))
print(batch)

# print(len(train_dataset))


# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader_small = DataLoader(small_test_dataset, batch_size=batch_size, shuffle=False)
# test_loader_large = DataLoader(large_test_dataset, batch_size=batch_size, shuffle=False)


# wandb.finish()
#
#
