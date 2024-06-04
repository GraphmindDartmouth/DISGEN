import argparse
import torch
import os
import datapreprocessing
import numpy as np
import random
import wandb

import util_new

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import MoleculeNet
from graphsst2_dataset import get_dataset
from torch_geometric.data import DataLoader
from model import StructuralGCN, BaselineGCN
from train_util import train_step, eval_step, train_data


torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default="bbbp", choices=["bbbp", "hiv", "graphsst2"], help='datasets to consider')
parser.add_argument('--patience', default=50, type=int, help='patience for early stopping')
parser.add_argument('--model', default="BaselineGCN", choices=["StructuralGCN", "BaselineGCN"],
                    help='models to consider')
parser.add_argument('--dataset_root', default="/scratch/dataset",
                    help='datasets root')

parser.add_argument('--train_val_test_idx_save_root',
                    default="/scratch/saved_idx",
                    help='root to save train val test idx')


parser.add_argument('--structural_feature_name', default="eigen_vector",
                    choices=['eigen_vector', 'degree'])

parser.add_argument('--batch_size', default=32, type=int, help='patience for early stopping')
parser.add_argument('--gpu_id', default="0",  help='GPU id')


config_wandb = {}
# wandb.init(project="project_name", config=config_wandb)
wandb.init(mode="disabled")

args = parser.parse_args()

dataset_root = args.dataset_root
train_val_test_idx_save_address = args.train_val_test_idx_save_root

batch_size = args.batch_size
# name: "hiv", "bbbp", "graphsst2"
dataset_name = args.dataset_name
structural_feature_name = args.structural_feature_name

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print("Working on data: ", dataset_name)
if dataset_name == "bbbp":
    cur_data = PygGraphPropPredDataset(name="ogbg-molbbbp", pre_transform=None, root=dataset_root)
    dataset_root = dataset_root + "_bbbp"

elif dataset_name == "hiv":
    cur_data = MoleculeNet(root=dataset_root, name="HIV")
    dataset_root = dataset_root + "_hiv"

elif dataset_name == "graphsst2":
    cur_data = get_dataset(dataset_dir=dataset_root, dataset_name='Graph-SST2')  # 70,042 , 2 label, feature: yes
    dataset_root = dataset_root + "_graphsst2"

else:
    raise NotImplementedError("Dataset name error")


print("You are using: ", args.structural_feature_name)
# train_idx, val_idx, small_test_idx, large_test_idx = quick_test_datapreprocessing.quick_test_data_prep(cur_data)


train_val_test_idx_save_address = os.path.join(train_val_test_idx_save_address, dataset_name)

silo_train_idx, silo_val_idx, small_test_idx, _, large_test_idx = \
        fedavg_util.load_idx_files(train_val_test_idx_save_address)
train_idx = fedavg_util.train_val_idx_concat(silo_train_idx)
val_idx = fedavg_util.train_val_idx_concat(silo_val_idx)
print(len(train_idx), len(val_idx), len(small_test_idx), len(large_test_idx))
train_dataset = cur_data[train_idx]
# fedavg_util.draw_silo_stat(train_dataset)

val_dataset = cur_data[val_idx]
small_test_dataset = cur_data[small_test_idx]
fedavg_util.draw_silo_stat(small_test_dataset)

large_test_dataset = cur_data[large_test_idx]
fedavg_util.draw_silo_stat(large_test_dataset)


print("Starting upsampling and constructing structural feature...")
if dataset_name != "graphsst2":
    print("Starting upsampling for data: {}".format(dataset_name))
    train_dataset = quick_test_datapreprocessing.UpsampledDataset(train_dataset, dataset_root, dataset_name,
                                                                  structural_feature_name)
train_dataset = quick_test_datapreprocessing.StructuralFeatureDataset(train_dataset, dataset_root, "train",
                                                                      structural_feature_name)
val_dataset = quick_test_datapreprocessing.StructuralFeatureDataset(val_dataset, dataset_root, "val",
                                                                    structural_feature_name)
small_test_dataset = quick_test_datapreprocessing.StructuralFeatureDataset(small_test_dataset, dataset_root,
                                                                           "small_test", structural_feature_name)
large_test_dataset = quick_test_datapreprocessing.StructuralFeatureDataset(large_test_dataset, dataset_root,
                                                                           "large_test", structural_feature_name)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader_small = DataLoader(small_test_dataset, batch_size=batch_size, shuffle=False)
test_loader_large = DataLoader(large_test_dataset, batch_size=batch_size, shuffle=False)

x_s = train_dataset[0].x.shape[-1]
y_s = train_dataset[0].y.shape[-1]

model_name = args.model
if model_name == "StructuralGCN":
    model = StructuralGCN(x_s, y_s)
elif model_name == "BaselineGCN":
    model = BaselineGCN(x_s, y_s)
else:
    raise NotImplementedError("Model not implemented!")

wandb.watch(model, log='all')
# model = StructuralGCN(x_s, y_s)
# model = BaselineGCN(x_s, y_s)

cls_criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loss = train_step(train_loader, model, optimizer, cls_criterion, device)
# print("Train loss: ", train_loss)
# print("Eval: ", eval_step(val_loader, model, cls_criterion, device, compute_loss=False))
train_data(train_loader, val_loader, test_loader_small, test_loader_large, model, optimizer, device)

wandb.finish()

