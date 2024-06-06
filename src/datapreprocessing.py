import os
import pickle

import numpy as np
import networkx as nx
import torch
import util_new

import aug_util

from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import InMemoryDataset, Data


class UpsampledDataset(InMemoryDataset):
    """
    Address the data imbalance issues
    """

    def __init__(self, cur_dataset, dataset_root, dataset_name, structural_feature_name, transform=None, pre_transform=None, pre_filter=None):

        self.cur_dataset = cur_dataset
        if structural_feature_name == "degree":
            root_name = dataset_root + "_structural_3Hops_upsample_train"
        elif structural_feature_name == "eigen_vector":
            root_name = dataset_root + "_structural_eigen_3Hops_upsample_train"
        elif structural_feature_name == "ori":
            root_name = dataset_root + "_ori_upsample_train"
        else:
            raise NotImplementedError("Error in upsampling")

        root = os.path.join(root_name, "train")
        self.cur_dataset = cur_dataset
        self.dataset_name = dataset_name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    # def __len__(self) -> int:
    #     return len(self.label)

    def process(self):
        data_list = []

        for idx, sample in enumerate(self.cur_dataset):
            if self.dataset_name == "bbbp":
                curr_y = sample.y.item()
                if curr_y == 1:
                    data_list.append(sample)
                elif curr_y == 0:
                    for _ in range(6):
                        data_list.append(sample)
                else:
                    raise NotImplementedError("Check your dataset!")

            elif self.dataset_name == "bace":
                curr_y = sample.y.item()
                if curr_y == 0:
                    data_list.append(sample)
                elif curr_y == 1:
                    for _ in range(2):
                        data_list.append(sample)
                else:
                    raise NotImplementedError("Check your dataset!")

            elif self.dataset_name == "hiv":
                curr_y = sample.y.item()
                if curr_y == 0:
                    data_list.append(sample)
                elif curr_y == 1:
                    for _ in range(20):
                        data_list.append(sample)
                else:
                    raise NotImplementedError("Check your dataset!")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def create_structural_feature(sample, feature_name):
    G = to_networkx(sample, to_undirected=True)
    nodes = G.nodes()
    degree = G.degree()
    features = []
    if feature_name == "degree":
        for node in nodes:
            # print(node)
            curr_node_feature = []
            subgraph = nx.ego_graph(G, node, undirected=True, radius=3)
            subgraph_degree_closeness = nx.degree_centrality(subgraph)
            closeness_score = subgraph_degree_closeness[node]
            curr_node_feature.append(closeness_score)

            # print(subgraph_degree_closeness)
            one_hop_nodes = nx.ego_graph(G, node, undirected=True, radius=1).nodes()
            one_hop_closeness = [subgraph_degree_closeness[i] for i in one_hop_nodes]
            curr_node_feature.append(np.max(one_hop_closeness))
            curr_node_feature.append(np.min(one_hop_closeness))
            curr_node_feature.append(np.average(one_hop_closeness))
            curr_node_feature.append(np.std(one_hop_closeness))

            features.append(curr_node_feature)
    elif feature_name == "eigen_vector":
        for node in nodes:
            # print(node)
            curr_node_feature = []
            subgraph = nx.ego_graph(G, node, undirected=True, radius=3)
            subgraph_eigen_closeness = nx.eigenvector_centrality(subgraph, max_iter=300, tol=1.0e-3)
            closeness_score = subgraph_eigen_closeness[node]
            curr_node_feature.append(closeness_score)

            # print(subgraph_degree_closeness)
            one_hop_nodes = nx.ego_graph(G, node, undirected=True, radius=1).nodes()
            one_hop_closeness = [subgraph_eigen_closeness[i] for i in one_hop_nodes]
            curr_node_feature.append(np.max(one_hop_closeness))
            curr_node_feature.append(np.min(one_hop_closeness))
            curr_node_feature.append(np.average(one_hop_closeness))
            curr_node_feature.append(np.std(one_hop_closeness))
            features.append(curr_node_feature)
    else:
        raise NotImplementedError("Error in create_structural_feature")

    features = torch.tensor(features, dtype=torch.float32)
    # print(features)
    return features





def save_idx_files(train_val_test_idx_save_address, train_idx, val_idx, small_test_idx, large_test_idx):
#     if not os.path.exists(train_val_test_idx_save_address):
#         os.makedirs(train_val_test_idx_save_address)
    pickle.dump(train_idx, open(f"{train_val_test_idx_save_address}/train_idx.pkl", "wb"))
    pickle.dump(val_idx, open(f"{train_val_test_idx_save_address}/val_idx.pkl", "wb"))
    pickle.dump(small_test_idx, open(f"{train_val_test_idx_save_address}/small_test_idx.pkl", "wb"))
    pickle.dump(large_test_idx, open(f"{train_val_test_idx_save_address}/large_test_idx.pkl", "wb"))
    print("Index file save success! ")


def quick_test_data_prep(cur_data):

    sizes = []
    for sample in cur_data:
        sizes.append(sample.x.shape[0])

    sizes = np.array(sizes)
    sorted_index = np.argsort(sizes)  # small to large

    num_test = np.ceil(sizes.shape[0]*0.10).astype(int)
    num_train_val_test = np.ceil(sizes.shape[0]*0.50).astype(int)

    large_test_idx = sorted_index[-num_test:]
    train_val_test_idx = sorted_index[:num_train_val_test]

    train_split, val_split, test_split = 0.7, 0.15, 0.15

    ones_idx_list, zeros_idx_list = [], []
    for idx in range(len(cur_data)):
        if idx in train_val_test_idx and cur_data[idx].x.shape[0] != 1:
            if cur_data[idx].y.item() == 1:
                ones_idx_list.append(idx)
            elif cur_data[idx].y.item() == 0.0:
                zeros_idx_list.append(idx)
            else:
                raise NotImplementedError("Error in binary check if it is bbbp or bace or PROTEINS!")

    ones_idx = np.array(ones_idx_list)
    zeros_idx = np.array(zeros_idx_list)

    ones_idx_perm = np.random.permutation(ones_idx)
    zeros_idx_perm = np.random.permutation(zeros_idx)
    lens_ones = len(ones_idx_list)
    lens_zeros = len(zeros_idx_list)

    num_train_ones, num_test_ones = int(lens_ones * train_split), int(lens_ones * test_split)
    num_train_zeros, num_test_zeros = int(lens_zeros * train_split), int(lens_zeros * test_split)

    train_one_idx, test_one_idx, val_one_idx = ones_idx_perm[:num_train_ones], \
        ones_idx_perm[num_train_ones:num_test_ones + num_train_ones], ones_idx_perm[num_test_ones + num_train_ones:]
    train_zero_idx, test_zero_idx, val_zero_idx = zeros_idx_perm[:num_train_zeros], \
        zeros_idx_perm[num_train_zeros:num_test_zeros + num_train_zeros], zeros_idx_perm[num_test_zeros + num_train_zeros:]

    print(f"train zero {len(train_zero_idx)}, train one {len(train_one_idx)}")
    print(f"test zero {len(test_zero_idx)}, test one {len(test_one_idx)}")
    print(f"val zero {len(val_zero_idx)}, val one {len(val_one_idx)}")

    train_idx = np.random.permutation(np.concatenate([train_one_idx, train_zero_idx]))
    val_idx = np.random.permutation(np.concatenate([val_one_idx, val_zero_idx]))
    small_test_idx = np.random.permutation(np.concatenate([test_one_idx, test_zero_idx]))

    large_test_idx_recorder = []
    num_large_zeros, num_large_ones = 0, 0
    cap_large_zero = len(test_zero_idx)
    cap_large_one = len(test_one_idx)

    for i, idx in enumerate(np.flip(sorted_index)):
        sample = cur_data[idx]
        if sample.y.item() == 0 and num_large_zeros < cap_large_zero:
            large_test_idx_recorder.append(idx)
            num_large_zeros += 1
        elif sample.y.item() == 1 and num_large_ones < cap_large_one:
            num_large_ones += 1
            large_test_idx_recorder.append(idx)
        elif sample.y.item() != 0 and sample.y.item() != 1:
            raise NotImplementedError("Check dataset y entries!")
        if num_large_zeros >= cap_large_zero and num_large_ones >= cap_large_one:
            print(
                f"Finish finding the appropriate samples, cost {i} searches for {len(large_test_idx_recorder)} "
                f"samples out of total largest size {len(cur_data)} samples!")
            break
    print(cap_large_one, cap_large_zero)
    large_test_idx = np.array(large_test_idx_recorder)
    large_test_idx = np.random.permutation(large_test_idx)

    return train_idx, val_idx, small_test_idx, large_test_idx


class StructuralFeatureDataset(InMemoryDataset):
    def __init__(self, cur_dataset, dataset_root, split_name, structural_feature_name,
                 transform=None, pre_transform=None, pre_filter=None):

        self.cur_dataset = cur_dataset
        if structural_feature_name == "degree":
            root_name = dataset_root + "_structural_3Hops"
        elif structural_feature_name == "eigen_vector":
            root_name = dataset_root + "_structural_eigen_3Hops"
        else:
            raise NotImplementedError("Error in StructuralFeatureDataset root")
        root = os.path.join(root_name, split_name)

        self.structural_feature_name = structural_feature_name

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    def process(self):
        data_list = []

        for sample in self.cur_dataset:
            feature = create_structural_feature(sample, self.structural_feature_name)
            data_list.append(Data(x=sample.x, edge_index=sample.edge_index, y=sample.y,
                                  num_nodes=sample.num_nodes, closenes_feature=feature))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class AugmentedDatasetOri(InMemoryDataset):
    def __init__(self, cur_dataset, dataset_root, split_name, device, transform=None, pre_transform=None, pre_filter=None):
        self.cur_dataset = cur_dataset
        root_name = dataset_root + "_augmented_ori"
        root = os.path.join(root_name, split_name)
        self.device = device

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    def process(self):
        data_list = []

        for sample in self.cur_dataset:
            data_list.append(Data(x=sample.x, edge_index=sample.edge_index, y=sample.y, num_nodes=sample.num_nodes))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class AugmentedDatasetLabelChange(InMemoryDataset):
    def __init__(self, cur_dataset, dataset_root, split_name, model, cls_criterion, device, transform=None, pre_transform=None, pre_filter=None):
        self.cur_dataset = cur_dataset
        root_name = dataset_root + "_augmented_label_change"
        root = os.path.join(root_name, split_name)
        self.model = model
        self.cls_criterion = cls_criterion
        self.device = device

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    def process(self):
        data_list = []

        for sample in self.cur_dataset:
            aug_graph_label_change = aug_util.graph_aug(self.model, sample, self.cls_criterion,
                                                               self.device, 1)

            data_list.append(Data(x=aug_graph_label_change.x, edge_index=aug_graph_label_change.edge_index,
                                  y=sample.y, num_nodes=aug_graph_label_change.num_nodes))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class AugmentedDatasetLabelUnChange(InMemoryDataset):
    def __init__(self, cur_dataset, dataset_root, split_name, model, cls_criterion, device, transform=None, pre_transform=None, pre_filter=None):
        self.cur_dataset = cur_dataset
        root_name = dataset_root + "_augmented_label_unchange"
        root = os.path.join(root_name, split_name)
        self.model = model
        self.cls_criterion = cls_criterion
        self.device = device

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    def process(self):
        data_list = []

        for sample in self.cur_dataset:
            aug_graph_label_unchange = aug_util.graph_aug(self.model, sample, self.cls_criterion,
                                                                 self.device, 0)

            data_list.append(Data(x=aug_graph_label_unchange.x, edge_index=aug_graph_label_unchange.edge_index,
                                  y=sample.y, num_nodes=aug_graph_label_unchange.num_nodes))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

#
# class PairDataset(InMemoryDataset):
#     def __init__(self, datasetA, datasetB, datasetC, transform=None, pre_transform=None, pre_filter=None):
#         root = None
#         self.datasetA = datasetA
#         self.datasetB = datasetB
#         self.datasetC = datasetC
#
#         super().__init__(root, transform, pre_transform, pre_filter)
#
#     def __getitem__(self, idx):
#         return self.datasetA[idx], self.datasetB[idx], self.datasetC[idx]


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_ori':
            return self.x_ori.size(0)
        if key == 'edge_index_label_change':
            return self.x_label_change.size(0)
        if key == 'edge_index_label_unchange':
            return self.x_label_unchange.size(0)
        return super().__inc__(key, value, *args, **kwargs)

