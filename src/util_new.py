import math
import torch
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from model import StructuralGCN, BaselineGCN


def draw_distribution(data_distribution_list):
    print("Drawing...")
    # create histogram
    plt.hist(data_distribution_list)

    # display histogram
    plt.show()


def check_silo_data(silo_data):
    print("Checking...")
    # print("If directed: ", cur_data.is_directed())
    size_list = []
    label_list = []
    for i in trange(len(silo_data)):
        sample = silo_data[i]
        cur_size = sample.x.shape[0]
        cur_label = sample.y
        label_list.append(int(cur_label[0]))
        # print("Graph id: ", i, "size: ", cur_size)
        size_list.append(cur_size)

    return size_list, label_list


def draw_stat(silo_data):
    silo_size_distribution, silo_label_distribution = check_silo_data(silo_data)
    draw_distribution(silo_size_distribution)
    draw_distribution(silo_label_distribution)


def find_size_idx(cur_data, idx_list, size):
    found_size_idx = 0
    for idx, data_idx in enumerate(idx_list):
        if cur_data[data_idx].x.shape[0] == size:
            found_size_idx = idx
            break
    return found_size_idx


def load_idx_files(train_val_test_save_address):
    with open(f"{train_val_test_save_address}/silo_train_idx.pkl", "rb") as file:
        silo_train_idx = pickle.load(file)
    with open(f"{train_val_test_save_address}/silo_val_idx.pkl", "rb") as file:
        silo_val_idx = pickle.load(file)

    with open(f"{train_val_test_save_address}/small_test_idx.pkl", "rb") as file:
        small_test_idx = pickle.load(file)

    with open(f"{train_val_test_save_address}/val_server_idx.pkl", "rb") as file:
        val_server_idx = pickle.load(file)

    with open(f"{train_val_test_save_address}/large_test_idx.pkl", "rb") as file:
        large_test_idx = pickle.load(file)

    return silo_train_idx, silo_val_idx, small_test_idx, val_server_idx, large_test_idx


def load_iid_idx_files(train_val_test_save_address):
    with open(f"{train_val_test_save_address}/silo_train_idx.pkl", "rb") as file:
        silo_train_idx = pickle.load(file)
    with open(f"{train_val_test_save_address}/silo_val_idx.pkl", "rb") as file:
        silo_val_idx = pickle.load(file)

    return silo_train_idx, silo_val_idx


def split_more_silos(train_val_test_save_address, cur_data, train_val_test_save_address_more_silo):
    silo_train_idx, silo_val_idx, small_test_idx, large_test_idx = load_idx_files(train_val_test_save_address)

    if not os.path.exists(train_val_test_save_address_more_silo):
        os.makedirs(train_val_test_save_address_more_silo)

    if os.path.isfile(train_val_test_save_address_more_silo+"/more_silo_train_idx.pkl"):
        print("silo_more idx files exist, loading...")
        with open(f"{train_val_test_save_address_more_silo}/more_silo_train_idx.pkl", "rb") as file:
            silo_more_train_idx = pickle.load(file)
        with open(f"{train_val_test_save_address_more_silo}/more_silo_val_idx.pkl", "rb") as file:
            silo_more_val_idx = pickle.load(file)
    else:
        print("No silo_more idx files, start spliting....")
        silo_more_train_idx = split_cur_silos(silo_train_idx, cur_data)
        silo_more_val_idx = split_cur_silos(silo_val_idx, cur_data)
        pickle.dump(silo_more_train_idx, open(f"{train_val_test_save_address_more_silo}/more_silo_train_idx.pkl", "wb"))
        pickle.dump(silo_more_val_idx, open(f"{train_val_test_save_address_more_silo}/more_silo_val_idx.pkl", "wb"))

    silo_more_train_data = []
    for i in range(len(silo_train_idx)):
        cur_silo_data = cur_data[silo_train_idx[i]]
        if i == 0:
            silo_more_train_data.append(cur_silo_data[silo_more_train_idx[0]])
        elif i == 1:
            silo_more_train_data.append(cur_silo_data[silo_more_train_idx[1]])
            silo_more_train_data.append(cur_silo_data[silo_more_train_idx[2]])
        elif i == 2:
            silo_more_train_data.append(cur_silo_data[silo_more_train_idx[3]])
            silo_more_train_data.append(cur_silo_data[silo_more_train_idx[4]])

    silo_more_val_data = []
    for i in range(len(silo_train_idx)):
        cur_silo_data = cur_data[silo_val_idx[i]]
        if i == 0:
            silo_more_val_data.append(cur_silo_data[silo_more_val_idx[0]])
        elif i == 1:
            silo_more_val_data.append(cur_silo_data[silo_more_val_idx[1]])
            silo_more_val_data.append(cur_silo_data[silo_more_val_idx[2]])
        elif i == 2:
            silo_more_val_data.append(cur_silo_data[silo_more_val_idx[3]])
            silo_more_val_data.append(cur_silo_data[silo_more_val_idx[4]])

    return silo_more_train_data, silo_more_val_data, small_test_idx, large_test_idx


def split_cur_silos(silo_data_idx, cur_data):

    more_silos_all_data = []
    for i in range(len(silo_data_idx)):
        more_silos_ones = []
        more_silos_zeros = []

        ones_idx_list, zeros_idx_list = [], []
        cur_data_idx = silo_data_idx[i]
        cur_silo_data = cur_data[cur_data_idx]
        # sort first
        sizes = []
        for sample in cur_silo_data:
            sizes.append(sample.x.shape[0])
        sizes = np.array(sizes)
        sorted_index = np.argsort(sizes)

        for idx in sorted_index:
            if cur_silo_data[idx].y.item() == 1:
                ones_idx_list.append(idx)

            elif cur_silo_data[idx].y.item() == 0.0:
                zeros_idx_list.append(idx)

            else:
                raise NotImplementedError("Error in binary check if it is bbbp or bace or PROTEINS!")
        print("ones len: ", len(ones_idx_list), "zeros len: ", len(zeros_idx_list))
        # draw_silo_stat(cur_silo_data[ones_idx_list])
        # draw_silo_stat(cur_silo_data[zeros_idx_list])
        if i == 0:
            more_silos_all_data.append(np.random.permutation(np.concatenate([ones_idx_list, zeros_idx_list])))

        elif i == 1:
            # 11-14 15-18

            size_14_idx_one = find_size_idx(cur_silo_data, ones_idx_list, 14)
            more_silos_ones.append(ones_idx_list[0:size_14_idx_one])
            more_silos_ones.append(ones_idx_list[size_14_idx_one:])

            size_14_idx_zero = find_size_idx(cur_silo_data, zeros_idx_list, 14)
            more_silos_zeros.append(zeros_idx_list[0:size_14_idx_zero])
            more_silos_zeros.append(zeros_idx_list[size_14_idx_zero:])

            for j in range(2):
                more_silos_all_data.append(np.random.permutation(np.concatenate([more_silos_ones[j],
                                                                                 more_silos_zeros[j]])))
            # draw_silo_stat(cur_silo_data[more_silos_all_data[0]])
            # draw_silo_stat(cur_silo_data[more_silos_all_data[1]])
        elif i == 2:
            # 19-21 22-25
            size_22_idx_one = find_size_idx(cur_silo_data, ones_idx_list, 22)
            more_silos_ones.append(ones_idx_list[0:size_22_idx_one])
            more_silos_ones.append(ones_idx_list[size_22_idx_one:])

            size_22_idx_zero = find_size_idx(cur_silo_data, zeros_idx_list, 22)
            more_silos_zeros.append(zeros_idx_list[0:size_22_idx_zero])
            more_silos_zeros.append(zeros_idx_list[size_22_idx_zero:])

            for j in range(2):
                more_silos_all_data.append(np.random.permutation(np.concatenate([more_silos_ones[j],
                                                                                 more_silos_zeros[j]])))
            # draw_silo_stat(cur_silo_data[more_silos_all_data[2]])
            # draw_silo_stat(cur_silo_data[more_silos_all_data[3]])
        else:
            raise NotImplementedError("Wrong silo list length")

    return more_silos_all_data


def train_val_idx_concat(silo_idx):
    data_idx = np.concatenate([silo_idx[0], silo_idx[1], silo_idx[2]])
    return data_idx


def split_silo_data_into_iid(silo_idx_list, train_val_test_idx_save_address):
    all_silos_data = np.random.permutation(np.concatenate([silo_idx_list[i] for i in range(len(silo_idx_list))]))

    num_data = len(all_silos_data)
    print("Total data: ", num_data)
    num_data_each_silo = math.floor(num_data / len(silo_idx_list))

    splited_idx_list = []
    start_idx = 0
    end_idx = num_data_each_silo
    for i in range(len(silo_idx_list) - 1):
        silo_data_idx = all_silos_data[start_idx:end_idx]
        splited_idx_list.append(silo_data_idx)
        start_idx = end_idx
        end_idx += num_data_each_silo
    silo_data_idx = all_silos_data[start_idx:]
    splited_idx_list.append(silo_data_idx)

    pickle.dump(splited_idx_list, open(train_val_test_idx_save_address, "wb"))

    return splited_idx_list


def create_model_optimizer_list(silo_numbers, model_name, x_s, y_s):
    model_list = []
    optimizer_list = []
    criterion_list = []
    for i in range(silo_numbers):
        if model_name == "StructuralGCN":
            model = StructuralGCN(x_s, y_s)
        elif model_name == "BaselineGCN":
            model = BaselineGCN(x_s, y_s)
        else:
            raise NotImplementedError("Model not implemented!")

        model_list.append(model)

        cur_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer_list.append(cur_optimizer)
        criterion_list.append(torch.nn.BCEWithLogitsLoss())

    return model_list, optimizer_list, criterion_list

