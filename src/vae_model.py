import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torch_geometric.nn import GCNConv, global_mean_pool
from quick_test_model import BaselineGCN, StructuralGCN, MultiViewGCN, GinNet, PNANet, BaselineGraphTransformer


def mean_tensors(mean_1, mean_2, i):
    mean_1[i] = 0
    mean_2[i] = 1
    mean_t = torch.from_numpy(mean_1).float()
    mean_s = torch.from_numpy(mean_2).float()
    return mean_t, mean_s


def reparameterization(mean_t, mean_s, log_var_t, log_var_s):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    mean_1, mean_2 = mean_tensors(np.ones(16), np.zeros(16), 13)

    z1 = mean_t + torch.exp(log_var_t/2) @ torch.normal(mean_1.to(device), torch.eye(16).to(device))
    z2 = mean_s + torch.exp(log_var_s/2) @ torch.normal(mean_2.to(device), torch.eye(16).to(device))

    return z1, z2


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim_encoder):
        # input_dim: 16, hidden_dim_1: 32,  hidden_dim_2: 64, output_dim: 32
        super(MLPBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim_encoder = output_dim_encoder
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.lin1 = torch.nn.Linear(self.input_dim, self.hidden_dim_1)
        self.lin2 = torch.nn.Linear(self.hidden_dim_1, self.hidden_dim_2)

        self.encoder1_net = torch.nn.Linear(self.hidden_dim_2, self.output_dim_encoder)
        self.encoder2_net = torch.nn.Linear(self.hidden_dim_2, self.output_dim_encoder)

    def forward(self, input_feature):
        output = F.relu(self.lin1(input_feature))
        output = F.relu(self.lin2(output))
        output_1 = self.encoder1_net(output)
        output_2 = self.encoder2_net(output)
        return output_1, output_2


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim_encoder, hidden_dim_1, hidden_dim_2, z_dim):
        super(Encoder, self).__init__()

        self.z_dim = z_dim

        self.mlp_block = MLPBlock(input_dim, hidden_dim_1, hidden_dim_2, output_dim_encoder)

        # Output layers for each encoder
        self.mean_encoder_1 = nn.Linear(output_dim_encoder, z_dim)
        self.log_var_1 = nn.Linear(output_dim_encoder, z_dim)

        self.mean_encoder_2 = nn.Linear(output_dim_encoder, z_dim)
        self.log_var_2 = nn.Linear(output_dim_encoder, z_dim)

    def forward(self, input_feature):
        out_1, out_2 = self.mlp_block(input_feature)

        mean_t = self.mean_encoder_1(F.relu(out_1))
        log_var_t = self.log_var_1(F.relu(out_1))

        mean_s = self.mean_encoder_2(F.relu(out_2))
        log_var_s = self.log_var_2(F.relu(out_2))

        return mean_t, mean_s, log_var_t, log_var_s


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim, target_classes, sensitive_classes):
        super(Decoder, self).__init__()

        self.output_dim = output_dim

        self.decoder1_lin1 = torch.nn.Linear(z_dim, hidden_dim)
        self.decoder1_lin2 = torch.nn.Linear(hidden_dim, output_dim)

        self.decoder2_lin1 = torch.nn.Linear(z_dim, hidden_dim)
        self.decoder2_lin2 = torch.nn.Linear(hidden_dim, output_dim)

        self.target_lin = torch.nn.Linear(output_dim, target_classes)
        self.sensitive_lin = torch.nn.Linear(output_dim, sensitive_classes)

    def forward(self, z_1, z_2):
        out_1 = F.relu(self.decoder1_lin1(z_1))
        out_1 = F.relu(self.decoder1_lin2(out_1))
        y_pred = self.target_lin(out_1)

        # out_1 = z_1
        out_2 = z_2

        # out_1 = F.relu(self.decoder2_lin1(out_1))
        # out_1 = F.relu(self.decoder2_lin2(out_1))

        out_2 = F.relu(self.decoder2_lin1(out_2))
        out_2 = F.relu(self.decoder2_lin2(out_2))

        # s_zt = self.sensitive_lin(out_1)
        s_pred = self.sensitive_lin(out_2)

        # out_1_permuted = torch.vstack((out_1[-1], out_1[0:-1]))
        # out_1_concated = torch.hstack((out_1, out_1_permuted))

        # out_2_permuted = torch.vstack((out_2[-1], out_2[0:-1]))
        # out_2_concated = torch.hstack((out_2, out_2_permuted))
        # print(out_2_permuted.shape)
        # print(out_2_concated.shape)

        # s_zt = self.sensitive_lin(out_1_concated)
        # s_zs = self.sensitive_lin(out_2_concated)

        # return y_zt, s_zt, s_zs
        return y_pred, s_pred


class Vae(nn.Module):
    def __init__(self, input_dim, decoder_output_dim, output_dim_encoder, hidden_dim_1, hidden_dim_2, z_dim, target_classes, sensitive_classes):
        super(Vae, self).__init__()
        self.encoder = Encoder(input_dim, output_dim_encoder, hidden_dim_1, hidden_dim_2, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim_1, decoder_output_dim, target_classes, sensitive_classes)

    def forward(self, x):
        mean_t, mean_s, log_var_t, log_var_s = self.encoder(x)
        z1, z2 = reparameterization(mean_t, mean_s, log_var_t, log_var_s)
        # y_zt, s_zt, s_zs = self.decoder(z1, z2)
        y_pred, s_pred = self.decoder(z1, z2)

        # return (mean_t, mean_s, log_var_t, log_var_s), (y_zt, s_zt, s_zs), (z1, z2)
        return (mean_t, mean_s, log_var_t, log_var_s), (y_pred, s_pred), (z1, z2)


class GnnVae(nn.Module):
    def __init__(self, gnn_model_name, gnn_input_dim, gnn_output_dim, output_dim_encoder, output_dim_decoder, hidden_dim_1, hidden_dim_2, z_dim, target_classes, sensitive_classes):
        super(GnnVae, self).__init__()
        if gnn_model_name == "StructuralGCN":
            self.gnn = StructuralGCN(gnn_input_dim, gnn_output_dim)
        elif gnn_model_name == "BaselineGCN":
            self.gnn = BaselineGCN(gnn_input_dim, gnn_output_dim)
        elif gnn_model_name == "MultiViewGCN":
            self.gnn = MultiViewGCN(gnn_input_dim, gnn_output_dim)

        else:
            raise NotImplementedError("Model not implemented!")
        self.vae = Vae(gnn_output_dim, output_dim_decoder, output_dim_encoder, hidden_dim_1, hidden_dim_2, z_dim,
                       target_classes, sensitive_classes)

    def forward(self, x, edge_index, data):
        if self.training:
            mean_t_list = []
            mean_s_list = []
            log_var_t_list = []
            log_var_s_list = []
            z_1_list = []
            z_2_list = []

            x_ori, x_label_change, x_label_unchange = self.gnn(x, edge_index, data)
            vae_output = self.vae(x_ori)
            y_pred, s_pred = vae_output[1]
            z_1_ori, z_2_ori = vae_output[2]
            mean_t_ori, mean_s_ori, log_var_t_ori, log_var_s_ori = vae_output[0]

            mean_t_list.append(mean_t_ori)
            mean_s_list.append(mean_s_ori)
            log_var_t_list.append(log_var_t_ori)
            log_var_s_list.append(log_var_s_ori)
            z_1_list.append(z_1_ori)
            z_2_list.append(z_2_ori)


            vae_output_label_change = self.vae(x_label_change)
            y_pred_label_change, s_pred_label_change = vae_output_label_change[1]
            z_1_label_change, z_2_label_change = vae_output_label_change[2]

            mean_t_label_change, mean_s_label_change, log_var_t_label_change, log_var_s_label_change = vae_output_label_change[0]
            mean_t_list.append(mean_t_label_change)
            mean_s_list.append(mean_s_label_change)
            log_var_t_list.append(log_var_t_label_change)
            log_var_s_list.append(log_var_s_label_change)
            z_1_list.append(z_1_label_change)
            z_2_list.append(z_2_label_change)

            vae_output_label_unchange = self.vae(x_label_unchange)
            y_pred_label_unchange, s_pred_label_unchange = vae_output_label_unchange[1]
            z_1_label_unchange, z_2_label_unchange = vae_output_label_unchange[2]

            mean_t_label_unchange, mean_s_label_unchange, log_var_t_label_unchange, log_var_s_label_unchange = vae_output_label_unchange[0]

            mean_t_list.append(mean_t_label_unchange)
            mean_s_list.append(mean_s_label_unchange)
            log_var_t_list.append(log_var_t_label_unchange)
            log_var_s_list.append(log_var_s_label_unchange)
            z_1_list.append(z_1_label_unchange)
            z_2_list.append(z_2_label_unchange)

            return y_pred, s_pred, s_pred_label_change, s_pred_label_unchange, mean_t_list, mean_s_list, \
                log_var_t_list, log_var_s_list, z_1_list, z_2_list
        else:
            x_ori = self.gnn(x, edge_index, data)
            vae_output = self.vae(x_ori)
            y_pred, s_pred = vae_output[1]
            return y_pred


class DisentGNN(nn.Module):
    def __init__(self, gnn_model_name, gnn_backbone, gnn_input_dim, gnn_output_dim, z_dim, hidden_dim, target_classes,
                 sensitive_classes, deg=None):
        super(DisentGNN, self).__init__()

        self.deg = deg
        # deg used for PNA

        self.gnn_model_name = gnn_model_name
        if self.gnn_model_name == "StructuralGCN":
            self.gnn = StructuralGCN(gnn_input_dim, gnn_output_dim)
        elif self.gnn_model_name == "BaselineGCN":
            self.gnn = BaselineGCN(gnn_input_dim, gnn_output_dim, disentangle=True)

        elif self.gnn_model_name == "BaselineGraphTransformer":
            self.gnn = BaselineGraphTransformer(gnn_input_dim, gnn_output_dim, disentangle=True)

        elif self.gnn_model_name == "BaselineGIN":
            self.gnn = GinNet(gnn_input_dim, gnn_output_dim, disentangle=True)

        elif self.gnn_model_name == "BaselinePNA":
            num_layers = 3
            self.gnn = PNANet(gnn_input_dim, num_layers, gnn_output_dim, self.deg)

        elif self.gnn_model_name == "MultiViewGCN" and gnn_backbone == "PNA":
            self.gnn = MultiViewGCN(gnn_input_dim, gnn_output_dim, gnn_backbone, self.deg)

        elif self.gnn_model_name == "MultiViewGCN":
            self.gnn = MultiViewGCN(gnn_input_dim, gnn_output_dim, gnn_backbone)

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
                # z_2_ori = F.relu(self.mlp_z2(x_ori))

                # y_pred = self.target_lin(self.lin1(x_ori))

                y_pred = self.target_lin(self.lin1(z_1_ori))
                # s_pred = self.sensitive_lin(self.lin2(z_2_ori))
                return y_pred

            elif self.gnn_model_name == "BaselineGraphTransformer":
                batch = data.batch
                x_ori = self.gnn(x, edge_index, batch)

                z_1_ori = F.relu(self.mlp_z1(x_ori))
                # z_2_ori = F.relu(self.mlp_z2(x_ori))

                # y_pred = self.target_lin(self.lin1(x_ori))

                y_pred = self.target_lin(self.lin1(z_1_ori))
                # s_pred = self.sensitive_lin(self.lin2(z_2_ori))
                return y_pred

            elif self.gnn_model_name == "BaselineGIN":

                x_ori = self.gnn(x, edge_index, data)

                z_1_ori = F.relu(self.mlp_z1(x_ori))
                # z_2_ori = F.relu(self.mlp_z2(x_ori))

                # y_pred = self.target_lin(self.lin1(x_ori))

                y_pred = self.target_lin(self.lin1(z_1_ori))
                # s_pred = self.sensitive_lin(self.lin2(z_2_ori))
                return y_pred

            elif self.gnn_model_name == "BaselinePNA":

                x_ori = self.gnn(x, edge_index)

                z_1_ori = F.relu(self.mlp_z1(x_ori))
                # z_2_ori = F.relu(self.mlp_z2(x_ori))

                # y_pred = self.target_lin(self.lin1(x_ori))

                y_pred = self.target_lin(self.lin1(z_1_ori))
                # s_pred = self.sensitive_lin(self.lin2(z_2_ori))

                return y_pred

            elif self.gnn_model_name == "MultiViewGCN":
                # x_ori = self.gnn(x, edge_index, data)
                #
                # z_1_ori = F.relu(self.mlp_z1(x_ori))
                # # z_2_ori = F.relu(self.mlp_z2(x_ori))
                #
                # # y_pred = self.target_lin(self.lin1(x_ori))
                #
                # y_pred = self.target_lin(self.lin1(z_1_ori))

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

                # # pass z2_ori to nn1, let the output be a uniform distribution
                # # z_2_ori = self.layer_norm_3(z_2_ori)
                # output_z2_ori_nn1 = self.target_lin(self.lin1(z_2_ori))
                # output_z2_ori_nn1 = self.layer_norm_1(output_z2_ori_nn1)
                # # output_z2_ori_nn1 = self.batch_norm(output_z2_ori_nn1)
                #
                # # pass z1_ori to nn2, let the output be a uniform distribution
                # # z_1_ori = self.layer_norm_3(z_1_ori)
                # output_z1_ori_nn2 = self.sensitive_lin(self.lin2(z_1_ori))
                # output_z1_ori_nn2 = self.layer_norm_2(output_z1_ori_nn2)
                # # return y_pred

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







