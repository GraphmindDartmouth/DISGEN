import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import BaselineGCN, MultiViewGCN


class DisentGNN(nn.Module):
    def __init__(self, gnn_model_name, gnn_input_dim, gnn_output_dim, z_dim, hidden_dim, target_classes,
                 sensitive_classes, deg=None):
        super(DisentGNN, self).__init__()

        self.deg = deg

        self.gnn_model_name = gnn_model_name
        if self.gnn_model_name == "BaselineGCN":
            self.gnn = BaselineGCN(gnn_input_dim, gnn_output_dim, disentangle=True)

        elif self.gnn_model_name == "MultiViewGCN":
            self.gnn = MultiViewGCN(gnn_input_dim, gnn_output_dim)

        else:
            raise NotImplementedError("Model not implemented!")

        self.mlp_z1 = torch.nn.Linear(gnn_output_dim, z_dim)
        self.mlp_z2 = torch.nn.Linear(gnn_output_dim, z_dim)

        self.lin1 = torch.nn.Linear(z_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(z_dim, hidden_dim)
        self.target_lin = torch.nn.Linear(hidden_dim, target_classes)
        self.sensitive_lin = torch.nn.Linear(hidden_dim, sensitive_classes)

        self.layer_norm_1 = torch.nn.LayerNorm(target_classes, elementwise_affine=True)
        self.layer_norm_2 = torch.nn.LayerNorm(sensitive_classes, elementwise_affine=True)

        self.layer_norm_3 = torch.nn.LayerNorm(z_dim)

        self.batch_norm = torch.nn.BatchNorm1d(target_classes)

    def forward(self, x, edge_index, data):
        if self.training:
            if self.gnn_model_name == "BaselineGCN":
                x_ori = self.gnn(x, edge_index, data)
                z_1_ori = F.relu(self.mlp_z1(x_ori))

                y_pred = self.target_lin(self.lin1(z_1_ori))
                # s_pred = self.sensitive_lin(self.lin2(z_2_ori))
                return y_pred

            elif self.gnn_model_name == "MultiViewGCN":

                x_ori, x_label_change, x_label_unchange = self.gnn(x, edge_index, data)

                z_1_ori = F.leaky_relu(self.mlp_z1(x_ori))
                z_2_ori = F.leaky_relu(self.mlp_z2(x_ori))

                y_pred = self.target_lin(self.lin1(z_1_ori))
                s_pred = self.sensitive_lin(self.lin2(z_2_ori))

                z_1_label_change = F.leaky_relu(self.mlp_z1(x_label_change))
                z_2_label_change = F.leaky_relu(self.mlp_z2(x_label_change))
                s_pred_label_change = self.sensitive_lin(self.lin2(z_2_label_change))

                z_1_label_unchange = F.leaky_relu(self.mlp_z1(x_label_unchange))
                z_2_label_unchange = F.leaky_relu(self.mlp_z2(x_label_unchange))
                s_pred_label_unchange = self.sensitive_lin(self.lin2(z_2_label_unchange))
                y_pred_label_unchange = self.target_lin(self.lin1(z_1_label_unchange))

                # return y_pred, s_pred, s_pred_label_change, s_pred_label_unchange, output_z2_ori_nn1, output_z1_ori_nn2
                return y_pred, s_pred, s_pred_label_change, s_pred_label_unchange, z_1_ori, z_2_ori, z_1_label_change, \
                    z_2_label_change, z_1_label_unchange, z_2_label_unchange, y_pred_label_unchange

        else: # inference
            # baseline models
            if self.gnn_model_name != "MultiViewGCN":
                x_ori = self.gnn(x, edge_index, data.batch)
                z_1_ori = F.relu(self.mlp_z1(x_ori))
                y_pred = self.target_lin(self.lin1(z_1_ori))

            else: # multiview
                x_ori = self.gnn(x, edge_index, data)
                z_1_ori = F.relu(self.mlp_z1(x_ori))
                y_pred = self.target_lin(self.lin1(z_1_ori))

            return y_pred

