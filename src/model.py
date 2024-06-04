import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential, Linear
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_max_pool, PNAConv, TransformerConv
from torch_geometric.utils import degree


class StructuralGCN(nn.Module):
    def __init__(self, x_s, ys=1):
        super(StructuralGCN, self).__init__()

        print("You are doing GCNNet with Closeness Structural features")
        neuron = 64
        attention_num_feature = 5
        self.aggregator = "self_max"

        self.conv1 = GCNConv(x_s, neuron, cached=False)
        self.conv2 = GCNConv(neuron, neuron, cached=False)
        self.conv3 = GCNConv(neuron, neuron, cached=False)

        self.closeness_layer = torch.nn.Linear(attention_num_feature, 1)
        self.attention1 = torch.nn.Linear(neuron, 16)
        self.attention2 = torch.nn.Linear(16, ys)

    def reset_parameters(self):
        torch.nn.init.zeros_(self.weight)
        torch.nn.init.constant_(self.alpha, self.args.alpha)
        torch.nn.init.zeros_(self.beta)
        torch.nn.init.constant_(self.gamma, self.args.gamma)

    def forward(self, x, edge_index, data, edge_weight=None):

        # x = data.x.type(torch.float32)
        # edge_index = data.edge_index
        x = x.type(torch.float32)

        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight))

        # Attention on each input graph
        closeness_hand_feature = data.closenes_feature
        size = int(data.batch.max().item() + 1)
        batch = data.batch
        tensor_cat_helper = []
        # print("size: ", size)
        # print(closeness_hand_feature.shape)
        # Here simply we use naive attention
        for i in range(size):
            mask = torch.eq(batch, i)
            input_tensor = x[mask]
            input_closeness_feature = closeness_hand_feature[mask]
            curr_size = input_tensor.shape[0]

            # print(input_closeness_feature.dtype)
            # print(input_closeness_feature.shape)
            input_closeness_feature = self.closeness_layer(input_closeness_feature)
            # print(input_closeness_feature.shape)
            input_closeness_feature = F.softmax(input_closeness_feature, dim=0)
            # print(input_closeness_feature.min().item() * curr_size)
            # exit()
            # # 0 - 1
            # min_value = input_closeness_feature.min().item()
            # max_value = input_closeness_feature.max().item()
            # diff = max_value - min_value
            # if diff == 0 :
            #     print("diff 0!")
            # input_closeness_feature = (input_closeness_feature - min_value) / diff

            # input_closeness_feature = torch.unsqueeze(input_closeness_feature, dim=-1)
            # print("here!")

            if self.aggregator == "self_max":
                input_closeness_feature = input_closeness_feature * curr_size  # * self.alpha
                attentioned, _ = torch.max(input_closeness_feature * input_tensor, dim=0)
            elif self.aggregator == "self_average":
                input_closeness_feature = input_closeness_feature * curr_size  # * self.alpha
                attentioned = torch.mean(input_closeness_feature * input_tensor, dim=0)
            else:
                raise NotImplementedError("Error in aggregator")
            attentioned = F.relu(self.attention1(attentioned))
            output = self.attention2(attentioned)
            # print(output.shape)

            # exit()
            # output = self.high_alpha * high_output + self.low_alpha * low_output
            output = torch.unsqueeze_copy(output, dim=0)
            tensor_cat_helper.append(output)

        x = torch.vstack(tensor_cat_helper)
        # print(x.shape)
        # x = torch.unsqueeze_copy(x, dim=-1)
        # print(x)
        # print(x.shape)
        # exit()
        return x


# class BaselineGraphTransformer(nn.Module):
#     def __init__(self, x_s, ys=1, disentangle=False):
#         super(BaselineGraphTransformer, self).__init__()
#
#         print("You are doing GT")
#         neuron = 64
#         self.disentangle = disentangle
#         self.conv1 = TransformerConv(x_s, neuron, cached=False)
#         self.conv2 = TransformerConv(neuron, neuron, cached=False)
#         self.conv3 = TransformerConv(neuron, neuron, cached=False)
#
#         self.lin1 = torch.nn.Linear(neuron, 16)
#         self.lin2 = torch.nn.Linear(16, ys)
#
#     def reset_parameters(self):
#         torch.nn.init.zeros_(self.weight)
#         torch.nn.init.constant_(self.alpha, self.args.alpha)
#         torch.nn.init.zeros_(self.beta)
#         torch.nn.init.constant_(self.gamma, self.args.gamma)
#
#     def forward(self, x, edge_index, batch, edge_weight=None):
#         # x = data.x.type(torch.float32)
#         # edge_index = data.edge_index
#         x = x.type(torch.float32)
#         edge_index = edge_index.type(torch.int64)
#         # edge_index = data.edge_index
#
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         x = F.relu(self.conv3(x, edge_index))
#         x = global_mean_pool(x, batch)
#         if self.disentangle:
#             return x
#         else:
#             x = self.lin2(self.lin1(x))
#             return x


class BaselineGCN(nn.Module):
    def __init__(self, x_s, ys=1, disentangle=False):
        super(BaselineGCN, self).__init__()

        # print("You are doing baseline GCN")
        neuron = 64
        self.disentangle = disentangle
        self.conv1 = GCNConv(x_s, neuron, cached=False)
        self.conv2 = GCNConv(neuron, neuron, cached=False)
        self.conv3 = GCNConv(neuron, neuron, cached=False)

        self.lin1 = torch.nn.Linear(neuron, 16)
        self.lin2 = torch.nn.Linear(16, ys)

    def reset_parameters(self):
        torch.nn.init.zeros_(self.weight)
        torch.nn.init.constant_(self.alpha, self.args.alpha)
        torch.nn.init.zeros_(self.beta)
        torch.nn.init.constant_(self.gamma, self.args.gamma)

    def forward(self, x, edge_index, data, edge_weight=None):
        # x = data.x.type(torch.float32)
        # edge_index = data.edge_index
        x = x.type(torch.float32)
        edge_index = edge_index.type(torch.int64)
        batch = data.batch
        # edge_index = data.edge_index

        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight))
        x = global_mean_pool(x, batch)
        if self.disentangle:
            return x
        else:
            x = self.lin2(self.lin1(x))
            return x



# class GinNet(nn.Module):
#     def __init__(self, x_s, ys=1, aggregator="average_pool", disentangle=False):
#         super(GinNet, self).__init__()
#         neuron = 64
#         self.disentangle = disentangle
#
#         r1 = np.random.uniform()
#         r2 = np.random.uniform()
#         r3 = np.random.uniform()
#
#         nn1 = Sequential(Linear(x_s, neuron))
#         self.conv1 = GINConv(nn1, eps=r1, train_eps=True)
#
#         nn2 = Sequential(Linear(neuron, neuron))
#         self.conv2 = GINConv(nn2, eps=r2, train_eps=True)
#
#         nn3 = Sequential(Linear(neuron, neuron))
#         self.conv3 = GINConv(nn3, eps=r3, train_eps=True)
#
#         self.fc1 = torch.nn.Linear(neuron, 10)
#         self.last = torch.nn.Linear(10, ys)
#
#         self.aggregator = aggregator
#
#         if self.aggregator == "average_pool":
#             self.agg = global_mean_pool
#         elif self.aggregator == "max_pool":
#             self.agg = global_max_pool
#
#     def reset_parameters(self):
#         torch.nn.init.zeros_(self.weight)
#         torch.nn.init.constant_(self.alpha, self.args.alpha)
#         # torch.nn.init.ones_(self.alpha)
#         torch.nn.init.zeros_(self.beta)
#         # torch.nn.init.ones_(self.gamma)
#         torch.nn.init.constant_(self.gamma, self.args.gamma)
#
#     def forward(self, x, edge_index, batch):
#
#         # x = data.x
#         # edge_index = data.edge_index
#         # batch = data.batch
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         x = F.relu(self.conv3(x, edge_index))
#
#         # if (self.aggregator != "average_pool") and (self.aggregator != "max_pool"):
#         #     if self.aggregator == "ASAP":
#         #         x, _, _, _, _ = self.agg(x, edge_index, batch=data.batch)
#         #
#         #     else:
#         #         x, _, _, _, _, _ = self.agg(x, edge_index, batch=data.batch)
#         # else:
#         #     x = self.agg(x, data.batch)
#         if self.disentangle:
#             x = self.agg(x, batch)
#         else:
#             x = self.agg(x, batch)
#             x = F.relu(self.fc1(x))
#             x = self.last(x)
#         return x



class MultiViewGCN(nn.Module):
    def __init__(self, gnn_input_dim, gnn_output_dim, deg=None):
        super(MultiViewGCN, self).__init__()

        neuron_1 = 32
        neuron_2 = 64
        self.gnn_backbone = "GCN"
        if self.gnn_backbone == "GCN":
            self.conv1 = GCNConv(gnn_input_dim, neuron_1, cached=False)
            self.conv2 = GCNConv(neuron_1, neuron_2, cached=False)
            self.conv3 = GCNConv(neuron_2, neuron_2, cached=False)

        else:
            raise NotImplementedError("Error in GNN backbone")

        # self.lin1 = torch.nn.Linear(neuron_2 * 2, 32)
        self.lin1 = torch.nn.Linear(neuron_2, 32)

        self.lin2 = torch.nn.Linear(32, gnn_output_dim)

    def reset_parameters(self):
        torch.nn.init.zeros_(self.weight)
        torch.nn.init.constant_(self.alpha, self.args.alpha)
        torch.nn.init.zeros_(self.beta)
        torch.nn.init.constant_(self.gamma, self.args.gamma)

    def forward(self, x, edge_index, data, edge_weight=None):
        # x = data.x.type(torch.float32)
        # edge_index = data.edge_index
        if self.training:
            x = x.type(torch.float32)
            edge_index = edge_index.type(torch.int64)
            batch = data.batch
            if self.gnn_backbone == "GCN":
                x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
                x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
                x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight))
                x = global_mean_pool(x, batch)
                # x_global_max = global_max_pool(x, batch)
                # x = self.lin2(self.lin1(torch.cat((x_global_mean, x_global_max), dim=1)))
                # x = self.lin2(self.lin1(x_global_mean))

                x_label_change_batch = data.x_label_change_batch
                x_aug_label_change = data.x_label_change.type(torch.float32)
                edge_index_aug_label_change = data.edge_index_label_change.type(torch.int64)
                # print(x_aug_label_change.shape, edge_index_aug_label_change.shape)
                x_1 = F.relu(self.conv1(x_aug_label_change, edge_index_aug_label_change, edge_weight=edge_weight))
                x_1 = F.relu(self.conv2(x_1, edge_index_aug_label_change, edge_weight=edge_weight))
                x_1 = F.relu(self.conv3(x_1, edge_index_aug_label_change, edge_weight=edge_weight))
                # x_1 = global_mean_pool(x_1, x_label_change_batch)
                x_1 = global_mean_pool(x_1, x_label_change_batch)
                # x_1_global_max = global_max_pool(x_1, x_label_change_batch)
                # # x_1 = self.lin2(self.lin1(torch.cat((x_1_global_mean, x_1_global_max), dim=1)))
                # x_1 = self.lin2(self.lin1(x_1_global_mean))

                x_label_unchange_batch = data.x_label_unchange_batch
                x_aug_label_unchange = data.x_label_unchange.type(torch.float32)
                edge_index_aug_label_unchange = data.edge_index_label_unchange.type(torch.int64)
                # print(x_aug_label_unchange.shape, edge_index_aug_label_unchange.shape)
                x_2 = F.relu(self.conv1(x_aug_label_unchange, edge_index_aug_label_unchange, edge_weight=edge_weight))
                x_2 = F.relu(self.conv2(x_2, edge_index_aug_label_unchange, edge_weight=edge_weight))
                x_2 = F.relu(self.conv3(x_2, edge_index_aug_label_unchange, edge_weight=edge_weight))
                x_2 = global_mean_pool(x_2, x_label_unchange_batch)
                # # x_2_global_max = global_max_pool(x_2, x_label_unchange_batch)
                # # x_2 = self.lin2(self.lin1(torch.cat((x_2_global_mean, x_2_global_max), dim=1)))
                # x_2 = self.lin2(self.lin1(x_2_global_mean))

                return x, x_1, x_2

            # elif self.gnn_backbone == "GT":
            #     x = F.relu(self.conv(x, edge_index, batch))
            #     # x_global_max = global_max_pool(x, batch)
            #     # x = self.lin2(self.lin1(torch.cat((x_global_mean, x_global_max), dim=1)))
            #     # x = self.lin2(self.lin1(x_global_mean))
            #
            #     x_label_change_batch = data.x_label_change_batch
            #     x_aug_label_change = data.x_label_change.type(torch.float32)
            #     edge_index_aug_label_change = data.edge_index_label_change.type(torch.int64)
            #     # print(x_aug_label_change.shape, edge_index_aug_label_change.shape)
            #     x_1 = F.relu(self.conv(x_aug_label_change, edge_index_aug_label_change, x_label_change_batch))
            #
            #     x_label_unchange_batch = data.x_label_unchange_batch
            #     x_aug_label_unchange = data.x_label_unchange.type(torch.float32)
            #     edge_index_aug_label_unchange = data.edge_index_label_unchange.type(torch.int64)
            #     # print(x_aug_label_unchange.shape, edge_index_aug_label_unchange.shape)
            #     x_2 = F.relu(self.conv(x_aug_label_unchange, edge_index_aug_label_unchange, x_label_unchange_batch))
            #
            #     return x, x_1, x_2
            #
            #
            # elif self.gnn_backbone == "GIN":
            #     x = F.relu(self.conv(x, edge_index, batch))
            #     # x = global_mean_pool(x, batch)
            #
            #     x_label_change_batch = data.x_label_change_batch
            #     x_aug_label_change = data.x_label_change.type(torch.float32)
            #     edge_index_aug_label_change = data.edge_index_label_change.type(torch.int64)
            #     # print(x_aug_label_change.shape, edge_index_aug_label_change.shape)
            #     x_1 = F.relu(self.conv(x_aug_label_change, edge_index_aug_label_change, x_label_change_batch))
            #
            #     # x_1 = global_mean_pool(x_1, x_label_change_batch)
            #     # x_1 = global_mean_pool(x_1, x_label_change_batch)
            #     # x_1_global_max = global_max_pool(x_1, x_label_change_batch)
            #     # # x_1 = self.lin2(self.lin1(torch.cat((x_1_global_mean, x_1_global_max), dim=1)))
            #     # x_1 = self.lin2(self.lin1(x_1_global_mean))
            #
            #     x_label_unchange_batch = data.x_label_unchange_batch
            #     x_aug_label_unchange = data.x_label_unchange.type(torch.float32)
            #     edge_index_aug_label_unchange = data.edge_index_label_unchange.type(torch.int64)
            #     # print(x_aug_label_unchange.shape, edge_index_aug_label_unchange.shape)
            #     x_2 = F.relu(self.conv(x_aug_label_unchange, edge_index_aug_label_unchange, x_label_unchange_batch))
            #     # x_2 = global_mean_pool(x_2, x_label_unchange_batch)
            #     # # x_2_global_max = global_max_pool(x_2, x_label_unchange_batch)
            #     # # x_2 = self.lin2(self.lin1(torch.cat((x_2_global_mean, x_2_global_max), dim=1)))
            #     # x_2 = self.lin2(self.lin1(x_2_global_mean))
            #     return x, x_1, x_2

        else:
            x = x.type(torch.float32)
            edge_index = edge_index.type(torch.int64)
            batch = data.batch
            # batch = batch

            if self.gnn_backbone == "GCN":
                x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
                x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
                x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight))
                x = global_mean_pool(x, batch)
                # x_global_max = global_max_pool(x, batch)
                # x = self.lin2(self.lin1(torch.cat((x_global_mean, x_global_max), dim=1)))
                # x = self.lin2(self.lin1(x_global_mean))

            elif self.gnn_backbone == "GIN":
                x = self.conv(x, edge_index, batch)
                # x = global_mean_pool(x, batch)

            elif self.gnn_backbone == "GT":
                x = self.conv(x, edge_index, batch)

            return x



