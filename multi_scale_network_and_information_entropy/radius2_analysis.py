import os
import glob
import time
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
import scipy.stats


def data_list(sample_path):
    sub_dirs = [x[0] for x in os.walk(sample_path)]
    sub_dirs.pop(0)

    data_list = []

    for sub_dir in sub_dirs:
        file_list = []
        dir_name = os.path.basename(sub_dir)
        file_glob = os.path.join(sample_path, dir_name, '*')
        file_list.extend(glob.glob(file_glob))

        for file_name in file_list:
            data_list.append([file_name, dir_name])

    return np.array(data_list)


class MRI_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, idx):
        filepath, target = self.data_list[idx][0], int(self.data_list[idx][1])
        dataframe = pd.read_csv(filepath, sep="\s+", header=None)
        pic = dataframe.to_numpy()

        return pic, target, idx

    def __len__(self):
        return len(self.data_list)


def main():
    parser = argparse.ArgumentParser(description="MDD")
    parser.add_argument('-R', '--sparserate', type=float, default=0.4, metavar='S')
    args = parser.parse_args()

    DATA_PATH = 'D:/ASUS/code/DTI_data/ADNI3_ADvsMCI_FN'
    filelist = data_list(DATA_PATH)
    dataset = MRI_Dataset(filelist)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    for data, target, idx in dataloader:
        x = data.numpy()
        y = target.numpy()
        idx = idx.numpy()

    # analysis
    connected_nodes = np.array([0,  9, 10, 16, 47, 54, 55, 62, 64, 70, 72, 82, 84, 86, 87, 88])
    x = x[:, connected_nodes, :][:, :, connected_nodes]
    x_square = np.matmul(x, x)
    x_square_17 = x_square[:, 5, 7]
    # x_square_17 = x_square[:, 3, 5]

    x0 = x_square_17[y == 0]
    x1 = x_square_17[y == 1]
    t_value, p_value = scipy.stats.ttest_ind(x0, x1, axis=0, equal_var=False, nan_policy='omit')

    neighbor_5 = x[:, 5, :]
    neighbor_7 = x[:, 7, :]
    mulsum = np.sum(neighbor_5 * neighbor_7, axis=1)
    mul = neighbor_5 * neighbor_7
    non_zero = np.sum(neighbor_5 * neighbor_7, axis=0)
    non_zero_idx = np.where(non_zero != 0)

    mul_0 = mul[:, 0]
    mul_10 = mul[:, 10]
    mul_12 = mul[:, 12]
    mul_13 = mul[:, 13]
    mul_15 = mul[:, 15]

    m0 = mul_12[y == 0]
    m1 = mul_12[y == 1]
    t_value, p_value = scipy.stats.ttest_ind(m0, m1, axis=0, equal_var=False, nan_policy='omit')

    print()

    long_path = np.zeros((90, 90))
    long_path[62, 88] = 0.0535
    long_path[88, 54] = 0.0535

    long_path[62, 84] = 0.0894
    long_path[84, 54] = 0.0894

    long_path = long_path + long_path.T
    df = pd.DataFrame(long_path)
    df.to_csv("D:/long_path.edge", sep='\t', header=False, index=False)


if __name__ == "__main__":
    main()
