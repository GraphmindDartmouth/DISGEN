import math
import torch
import pickle
import os

import matplotlib.pyplot as plt
from tqdm import trange


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


def draw_silo_stat(silo_data):
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





