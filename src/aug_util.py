import math
import torch
import torch_geometric

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import fedavg_util

from scipy.sparse import csr_array
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data


def graph_explainer(model, data, device):
    model.eval()

    # fig = plt.figure()
    # g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    # nx.draw(g)
    # plt.show()

    # data.edge_index = data.edge_index[:, ::2]
    # get half of the edge index in case gnn explainer generates directed edge mask

    for explanation_type in ['phenomenon']:
        explainer = Explainer(
            model=model,
            # algorithm=GNNExplainer(epochs=300),
            algorithm=PGExplainer(epochs=30).to(device),

            explanation_type=explanation_type,
            # node_mask_type='object',
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='raw',
            ),)
        model.to(device)
        for epoch in range(30):
            loss = explainer.algorithm.train(epoch, model, data.x, data.edge_index, target=data.y, data=data)

        # targets, preds = [], []
        target = data.y if explanation_type == 'phenomenon' else None
        explanation = explainer(data.x, data.edge_index, target=target, data=data)
        # print(f'Generated model explanations in {explanation.available_explanations}')
        # explanation.visualize_graph(backend="networkx")
        # print("Original edge index: ", data.edge_index)
        # print("Node mask: ", explanation["node_mask"])
        # print("Edge mask: ", explanation["edge_mask"])
        edge_mask = explanation["edge_mask"].data

        # edge_mask_undir = (edge_mask[0::2] + edge_mask[1::2]) / 2

        # edge_index_undir = edge_index[:, 0::2]
        # edge_mask_undir = edge_mask_undir.cpu().numpy()
        # edge_index_undir = edge_index_undir.cpu().numpy()

    return edge_mask


# func to calculate gradients for a small input graph
def cal_edge_grad(model, data, cls_criterion, device):
    model.eval()
    model.zero_grad()
    edge_index = data.edge_index
    edge_weight = torch.ones((edge_index.size(1)), device=device)
    edge_weight.requires_grad = True

    pred = model(data.x, edge_index, data, edge_weight)
    # grad_1 = torch.autograd.grad(pred, edge_index, create_graph=True)
    is_labeled = data.y == data.y
    loss = cls_criterion(pred.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled])
    loss.backward()
    # print("Edge grad: ", edge_weight.grad.data)
    # edge_grad = edge_weight.grad.data[0::2] + edge_weight.grad.data[1::2]
    edge_grad = edge_weight.grad.data
    # print("Final edge grad: ", edge_grad)
    # undir_edge_index = edge_index[:, 0::2]
    # print(undir_edge_index)
    # sorted_grad_idx = np.argsort(edge_grad.cpu())
    edge_grad = edge_grad.cpu().numpy()
    return edge_grad


def graph_aug_size_change_label_unchange(model, data, edge_mask, remove_rate, device):
    input_graph = to_networkx(data, to_undirected=True, node_attrs=['x'])

    # remove isolated nodes
    isolated_nodes = list(nx.isolates(input_graph))
    for i in isolated_nodes:
        input_graph.remove_node(i)
    cur_graph_no_isolated = input_graph

    cur_graph_pyg = from_networkx(cur_graph_no_isolated, group_node_attrs=['x'])
    cur_graph_nx = to_networkx(cur_graph_pyg, to_undirected=True, node_attrs=['x'])

    edge_index = cur_graph_pyg.edge_index

    row = edge_index[0, :].cpu().numpy()
    col = edge_index[1, :].cpu().numpy()
    csr_matrix = csr_array((edge_mask.cpu().numpy(), (row, col)), shape=(cur_graph_nx.number_of_nodes(),
                                                                         cur_graph_nx.number_of_nodes())).toarray()

    # node_aggregated_edge_mask_map = {}
    node_aggregated_edge_mask = []

    for node in cur_graph_nx.nodes:
        # print("Node: ", node)
        neighbor_list = [n for n in cur_graph_nx.neighbors(node)]
        aggregated_edge_mask = sum(csr_matrix[node, neighbor_list]) / len(neighbor_list)
        node_aggregated_edge_mask.append(aggregated_edge_mask)

    sorted_mask_idx = np.argsort(node_aggregated_edge_mask)

    while True:
        num_nodes_to_remove = math.ceil(remove_rate * len(sorted_mask_idx))
        node_idx_to_remove = sorted_mask_idx[:num_nodes_to_remove]
        # print("Removed node idx: ", node_idx_to_remove)
        for i in node_idx_to_remove:
            cur_graph_nx.remove_node(i)

        # fig = plt.figure()
        # nx.draw(cur_graph_nx)
        # plt.show()

        cur_graph_pyg_size_changed = from_networkx(cur_graph_nx, group_node_attrs=['x'])

        # print("After removal: ", cur_graph_pyg.x.shape)

        with torch.no_grad():
            pred = model(cur_graph_pyg_size_changed.x.to(device), cur_graph_pyg_size_changed.edge_index.to(device), data)

        hard_pred = (pred.sigmoid() > 0.5).int()

        hard_pred = hard_pred.cpu().numpy()
        # print("Prediction: ", hard_pred)
        y_true = data.y.cpu().numpy()
        # print("Data Y: ", y_true)
        if hard_pred == y_true or num_nodes_to_remove == 1:
            break
        else:
            remove_rate = remove_rate / 2
            cur_graph_nx = to_networkx(cur_graph_pyg, to_undirected=True, node_attrs=['x'])
            # print("Number of nodes: ", cur_graph.number_of_nodes())

    augmented_graph = Data(x=cur_graph_pyg_size_changed.x, edge_index=cur_graph_pyg_size_changed.edge_index)
    return augmented_graph


def graph_aug_size_unchange_label_change(data, edge_mask, remove_rate):
    # print("Running aug 1")

    edge_index = data.edge_index.cpu().numpy()
    edge_index = edge_index[:, 0::2]
    edge_mask = edge_mask.cpu().numpy()
    edge_mask = (edge_mask[0::2] + edge_mask[1::2]) / 2

    # print("Edge index: ", edge_index)
    # print("Edge mask: ", edge_mask)

    sorted_mask_idx = np.argsort(edge_mask)
    # print("Sorted edge mask: ", sorted_mask_idx)

    reserved_edge_idx = edge_index[:, sorted_mask_idx[:math.floor((1 - remove_rate) * len(edge_mask))]]
    # print("Reserved edge idx: ", reserved_edge_idx)
    # print(reserved_edge_idx)
    final_edge_index = np.hstack((reserved_edge_idx, [reserved_edge_idx[1], reserved_edge_idx[0]]))
    # print("Final: ", final_edge_index)
    final_edge_index = torch.from_numpy(final_edge_index).type(torch.int64)
    augmented_graph = Data(x=data.x, edge_index=final_edge_index)
    return augmented_graph


def graph_aug(model, data, cls_criterion, device, aug_method):
    remove_rate = 0.2

    data.device = device
    data.to(device)
    model.to(device)

    edge_mask = graph_explainer(model, data, device)
    # print(edge_mask)

    # if all edge masks are same, use grad info to get graph aug
    if len(set(edge_mask.data)) == 1:
        edge_mask = cal_edge_grad(model, data, cls_criterion, device)
        print("Same edge mask, compute edge grad: ", edge_mask)

    # fig = plt.figure()
    # g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    # nx.draw(g)
    # plt.show()

    if aug_method == 0:
        # 0: Size change Label unchange
        augmented_graph = graph_aug_size_change_label_unchange(model, data, edge_mask, remove_rate, device)

        # fig = plt.figure()
        # g = torch_geometric.utils.to_networkx(data, to_undirected=True)
        # nx.draw(g)
        # plt.show()
        #
        # fig = plt.figure()
        # g = torch_geometric.utils.to_networkx(augmented_graph, to_undirected=True)
        # nx.draw(g)
        # plt.show()

    elif aug_method == 1:
        # 1: Size unchange Label change
        augmented_graph = graph_aug_size_unchange_label_change(data, edge_mask, remove_rate)

        # fig = plt.figure()
        # g = torch_geometric.utils.to_networkx(data, to_undirected=True)
        # nx.draw(g)
        # plt.show()
        #
        # fig = plt.figure()
        # g = torch_geometric.utils.to_networkx(augmented_graph, to_undirected=True)
        # nx.draw(g)
        # plt.show()
    return augmented_graph


