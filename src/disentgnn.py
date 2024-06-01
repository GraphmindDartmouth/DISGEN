import argparse
import torch
import os
import quick_test_datapreprocessing
import numpy as np
import random
import fedavg_util

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import MoleculeNet
from graphsst2_dataset import get_dataset
from torch_geometric.data import DataLoader
# from quick_test_model import StructuralGCN, BaselineGCN
from quick_test_train_util import train_step_gnn_vae, eval_step, train_data_gnn_vae, train_data_disentgnn
from quick_test_vae_model import DisentGNN, BaselineGCN
import wandb


torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default="bbbp", choices=["bbbp", "hiv", "graphsst2"], help='datasets to consider')
parser.add_argument('--patience', default=50, type=int, help='patience for early stopping')
parser.add_argument('--model', default="MultiViewGCN", choices=["StructuralGCN", "BaselineGCN", "MultiViewGCN",
                                                                "BaselineGIN", "BaselineGraphTransformer"],
                    help='models to consider')

parser.add_argument('--dataset_root', default="/scratch/dataset",
                    help='datasets root')

parser.add_argument('--train_val_test_idx_save_root',
                    default="/scratch/saved_idx",
                    help='root to save train val test idx')


parser.add_argument('--structural_feature_name', default="eigen_vector",
                    choices=['eigen_vector', 'degree'])

parser.add_argument('--criterion_name', default="pair_loss_cos",
                    choices=['pair_loss_triplet', 'pair_loss', 'cos_loss', 'pair_loss_cos', 'pearson_loss', 'none'])

parser.add_argument('--vae', default=True, help='insert vae or not')
parser.add_argument('--gnn_backbone', choices=['GCN', 'GIN', 'GT'], default='GCN', help='gnn backbone')

parser.add_argument('--batch_size', default=32, type=int, help='patience for early stopping')
parser.add_argument('--gpu_id', default="0",  help='GPU id')

args = parser.parse_args()

config_wandb = {}
wandb.init(project="disent_multiview_reconstruction", config=config_wandb)
# close wandb
# wandb.init(mode="disabled")

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


# print("You are using: ", args.structural_feature_name)
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
# fedavg_util.draw_silo_stat(small_test_dataset)

large_test_dataset = cur_data[large_test_idx]
# fedavg_util.draw_silo_stat(large_test_dataset)


# print("Starting upsampling and constructing structural feature...")
# print("Starting upsampling...")

if dataset_name != "graphsst2":
    print("Starting upsampling for data: {}".format(dataset_name))
    train_dataset = quick_test_datapreprocessing.UpsampledDataset(train_dataset, dataset_root, dataset_name, "ori")
# train_dataset = quick_test_datapreprocessing.StructuralFeatureDataset(train_dataset, dataset_root, "train",
#                                                                       structural_feature_name)
# val_dataset = quick_test_datapreprocessing.StructuralFeatureDataset(val_dataset, dataset_root, "val",
#                                                                     structural_feature_name)
# small_test_dataset = quick_test_datapreprocessing.StructuralFeatureDataset(small_test_dataset, dataset_root,
#                                                                            "small_test", structural_feature_name)
# large_test_dataset = quick_test_datapreprocessing.StructuralFeatureDataset(large_test_dataset, dataset_root,
#                                                                            "large_test", structural_feature_name)

cls_criterion = torch.nn.BCEWithLogitsLoss()
explainer_model_name = "BaselineGCN"
x_s = train_dataset[0].x.shape[-1]
y_s = train_dataset[0].y.shape[-1]
explainer_model = BaselineGCN(x_s, y_s)

model_save_path = "/scratch/saved_model/"
if os.path.isfile(model_save_path + explainer_model_name + "_saved.pt"):
    explainer_model = torch.load(model_save_path + explainer_model_name + "_saved.pt")
    print("Model loaded!!")
else:
    torch.save(explainer_model, model_save_path + explainer_model_name + "_saved.pt")
    print("Model saved!!")

train_dataset_ori = quick_test_datapreprocessing.AugmentedDatasetOri(train_dataset, dataset_root, "train", device)
train_dataset_label_change = quick_test_datapreprocessing.AugmentedDatasetLabelChange(train_dataset, dataset_root,
                                                                                      "train", explainer_model,
                                                                                      cls_criterion,
                                                                                      device)

train_dataset_label_unchange = quick_test_datapreprocessing.AugmentedDatasetLabelUnChange(train_dataset, dataset_root,
                                                                                          "train", explainer_model,
                                                                                          cls_criterion,
                                                                                          device)

data_list = []
for i in range(len(train_dataset_ori)):
    # print(train_dataset_label_change[i].x.shape)
    # print(train_dataset_label_change[i].edge_index.shape)
    # print(train_dataset_label_unchange[i].x.shape)
    # print(train_dataset_label_unchange[i].edge_index.shape)
    data_list.append(quick_test_datapreprocessing.PairData(x=train_dataset_ori[i].x,
                                                           edge_index=train_dataset_ori[i].edge_index,
                                          y=train_dataset_ori[i].y,
                                          x_label_change=train_dataset_label_change[i].x,
                                          edge_index_label_change=train_dataset_label_change[i].edge_index,
                                          x_label_unchange=train_dataset_label_unchange[i].x,
                                          edge_index_label_unchange=train_dataset_label_unchange[i].edge_index))

train_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, follow_batch=['x_ori', 'x_label_change',
                                                                                        'x_label_unchange'])
# batch = next(iter(train_loader))
# print(batch)


# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader_small = DataLoader(small_test_dataset, batch_size=batch_size, shuffle=False)
test_loader_large = DataLoader(large_test_dataset, batch_size=batch_size, shuffle=False)

x_s = train_dataset[0].x.shape[-1] # x_s = 9
y_s = train_dataset[0].y.shape[-1]

gnn_input_dim = x_s
gnn_output_dim = 64

hidden_dim = 32
# hidden_dim = 16

z_dim = 16
# z_dim = 64

target_classes = y_s
sensitive_classes = 9
model_name = args.model
gnn_backbone = args.gnn_backbone
model = DisentGNN(model_name, gnn_backbone, gnn_input_dim, gnn_output_dim, z_dim, hidden_dim, target_classes, sensitive_classes)
print(model)

wandb.watch(model, log='all')

criterion_name = args.criterion_name
print("Criterion name: ", criterion_name)


optimizer_1 = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer_2 = torch.optim.Adam(model.vae.parameters(), lr=0.001)

# train_loss = train_step_gnn_vae(train_loader, model, optimizer_1, optimizer_1, 10, device)
# print("Train loss: ", train_loss)
# print("Eval: ", eval_step(val_loader, model, cls_criterion, device, compute_loss=False))
train_data_disentgnn(train_loader, val_loader, test_loader_small, test_loader_large, model, optimizer_1,
                     device, criterion_name)

wandb.finish()

