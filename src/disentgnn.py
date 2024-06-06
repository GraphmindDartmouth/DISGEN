import argparse
import torch
import os
import datapreprocessing
import numpy as np
import random
import util_new

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import MoleculeNet, TUDataset
from graphsst2_dataset import get_dataset
from torch_geometric.data import DataLoader
from train_util import train_data_disentgnn
from model_disentgnn import DisentGNN, BaselineGCN
import wandb
import pickle

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="bbbp", choices=["bbbp", "PROTEINS", "nci1", "graphsst2"], help='datasets to consider')
parser.add_argument('--patience', default=50, type=int, help='patience for early stopping')
parser.add_argument('--model', default="MultiViewGCN", choices=["MultiViewGCN", "BaselineGCN"],
                    help='models to consider')

parser.add_argument('--dataset_root', default="/scratch/dataset",
                    help='datasets root')

parser.add_argument('--train_val_test_idx_save_root',
                    default="/scratch/saved_idx",
                    help='root to save train val test idx')

parser.add_argument('--model_save_path', default="/scratch/saved_model/",
                    help='root to save train val test idx')


parser.add_argument('--criterion', default="pair_loss_cos",
                    choices=['pair_loss_triplet', 'pair_loss', 'cos_loss', 'pair_loss_cos', 'pearson_loss', 'none'])


parser.add_argument('--batch_size', default=32, type=int, help='patience for early stopping')
parser.add_argument('--gpu_id', default="0",  help='GPU id')

parser.add_argument('--gnn_output_dim', default=64,  help='gnn output dim')
parser.add_argument('--hidden_dim', default=32,  help='hidden dim')
parser.add_argument('--z_dim', default=16,  help='z dim')
parser.add_argument('--sensitive_classes', default="9",  help='sensitive classes')

parser.add_argument('--beta1', default=0.5,  help='similarity loss weight')
parser.add_argument('--beta2', default=1,  help='target loss weight')
parser.add_argument('--beta3', default=5e4,  help='decoupling loss weight')
parser.add_argument('--lr', default=0.001,  help='learning rate')


args = parser.parse_args()

config_wandb = {}
# wandb.init(project="disent_multiview_reconstruction", config=config_wandb)
# close wandb
wandb.init(mode="disabled")

dataset_root = args.dataset_root

train_val_test_idx_save_address = args.train_val_test_idx_save_root

batch_size = args.batch_size
dataset = args.dataset
structural_feature_name = args.structural_feature_name

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print("Working on data: ", dataset)
if dataset == "bbbp":
    cur_data = PygGraphPropPredDataset(name="ogbg-molbbbp", pre_transform=None, root=dataset_root)
    dataset_root = dataset_root + "_bbbp"

if dataset == "nci1":
    cur_data = TUDataset(root=dataset_root, name='NCI1', use_node_attr=True)
    dataset_root = dataset_root + "_nci1"

elif dataset == "PROTEINS":
    cur_data = TUDataset(root=dataset_root, name='PROTEINS', use_node_attr=True)
    dataset_root = dataset_root + "_PROTEINS"

elif dataset == "graphsst2":
    cur_data = get_dataset(dataset_dir=dataset_root, dataset_name='Graph-SST2')  # 70,042 , 2 label, feature: yes
    dataset_root = dataset_root + "_graphsst2"

else:
    raise NotImplementedError("Dataset name error")


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

train_dataset = cur_data[train_idx]

val_dataset = cur_data[val_idx]
small_test_dataset = cur_data[small_test_idx]

large_test_dataset = cur_data[large_test_idx]

if dataset != "graphsst2":
    print("Starting upsampling for data: {}".format(dataset))
    train_dataset = datapreprocessing.UpsampledDataset(train_dataset, dataset_root, dataset, "ori")


beta_set = [args.beta1, args.beta2, args.beta3]

cls_criterion = torch.nn.BCEWithLogitsLoss()
explainer_model_name = "BaselineGCN"
x_s = train_dataset[0].x.shape[-1]
y_s = train_dataset[0].y.shape[-1]
explainer_model = BaselineGCN(x_s, y_s)

model_save_path = args.model_save_path
if os.path.isfile(model_save_path + explainer_model_name + "_saved.pt"):
    explainer_model = torch.load(model_save_path + explainer_model_name + "_saved.pt")
    print("Model loaded!!")
else:
    torch.save(explainer_model, model_save_path + explainer_model_name + "_saved.pt")
    print("Model saved!!")

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
    data_list.append(datapreprocessing.PairData(x=train_dataset_ori[i].x,
                                                           edge_index=train_dataset_ori[i].edge_index,
                                          y=train_dataset_ori[i].y,
                                          x_label_change=train_dataset_label_change[i].x,
                                          edge_index_label_change=train_dataset_label_change[i].edge_index,
                                          x_label_unchange=train_dataset_label_unchange[i].x,
                                          edge_index_label_unchange=train_dataset_label_unchange[i].edge_index))

train_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, follow_batch=['x_ori', 'x_label_change',
                                                                                        'x_label_unchange'])
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader_small = DataLoader(small_test_dataset, batch_size=batch_size, shuffle=False)
test_loader_large = DataLoader(large_test_dataset, batch_size=batch_size, shuffle=False)

x_s = train_dataset[0].x.shape[-1] # x_s = 9
y_s = train_dataset[0].y.shape[-1]

gnn_input_dim = x_s
gnn_output_dim = args.gnn_output_dim

hidden_dim = args.hidden_dim

z_dim = args.z_dim

target_classes = y_s
sensitive_classes = args.sensitive_classes
model_name = args.model
model = DisentGNN(model_name, gnn_input_dim, gnn_output_dim, z_dim, hidden_dim, target_classes, sensitive_classes)
print(model)

wandb.watch(model, log='all')

criterion = args.criterion
print("Criterion: ", criterion)


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_data_disentgnn(train_loader, val_loader, test_loader_small, test_loader_large, model, optimizer,
                     device, criterion, beta_set)

wandb.finish()

