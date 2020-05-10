# main.py for MI learning
import os
import glob
import time
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ggmfit import *
from mutual_information import *
from PSO import *
from MI_learning import *


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
    parser.add_argument('-I', '--idx', type=int, default=0, metavar='I')
    parser.add_argument('-R', '--sparserate', type=float, default=0.2, metavar='S')
    args = parser.parse_args()

    DATA_PATH = "D:/code/DTI_data/network_FN/"
    NODE_PATH = "D:/code/DTI_data/network_distance/AAL_90_num.node"
    GRAPH_PATH = "D:/code/DTI_data/network_distance/grouplevel.edge"
    output_path = "D:/code/mutual_information_MDD_output/"
    filelist = data_list(DATA_PATH)
    dataset = MRI_Dataset(filelist)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    for data, target, idx in dataloader:
        x = data.numpy()
        y = target.numpy()
        idx = idx.numpy()
    
    node_df = pd.read_csv(NODE_PATH, sep=' ', header=None)
    region = node_df.iloc[:, 3].to_numpy()
    x1 = x[:, region==1, :][:, :, region==1]
    x2 = x[:, region==2, :][:, :, region==2]
    x3 = x[:, region==3, :][:, :, region==3]
    x4 = x[:, region==4, :][:, :, region==4]
    x5 = x[:, region==5, :][:, :, region==5]
    node1 = np.where(region==1)[0]
    node2 = np.where(region==2)[0]
    node3 = np.where(region==3)[0]
    node4 = np.where(region==4)[0]
    node5 = np.where(region==5)[0]

    x1 = np.tanh(x1 / 10)
    x2 = np.tanh(x2 / 10)
    x3 = np.tanh(x3 / 10)
    x4 = np.tanh(x4 / 10)
    x5 = np.tanh(x5 / 10)

    # ggraph = pd.read_csv(GRAPH_PATH, sep='\t', header=None).to_numpy()
    # g1 = np.matrix(ggraph[node1, :][:, node1])
    # g2 = np.matrix(ggraph[node2, :][:, node2])
    # g3 = np.matrix(ggraph[node3, :][:, node3])
    # g4 = np.matrix(ggraph[node4, :][:, node4])
    # g5 = np.matrix(ggraph[node5, :][:, node5])

    # graph mine
    sparse_rate = args.sparserate
    g1 = graph_mine(x1, y, sparse_rate)
    g2 = graph_mine(x2, y, sparse_rate)
    g3 = graph_mine(x3, y, sparse_rate)
    g4 = graph_mine(x4, y, sparse_rate)
    g5 = graph_mine(x5, y, sparse_rate)

    seed = 123456
    np.random.seed(seed)
    starttime = time.time()

    # MI learning
    k = 6

    # PSO parameters
    part_num = 30
    iter_num = 2000
    omega_max = 0.9
    omega_min = 0.4
    c1 = 2
    c2 = 2

    # 10-fold validation
    cv = 10
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    for idx, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        if not idx == args.idx:
            continue
        y_train = y[train_idx]
        y_test = y[test_idx]

        # region 1
        x_train = x1[train_idx]
        x_test = x1[test_idx]

        model = MI_learning(x_train, y_train, g1, k)
        b_list, MI_list = model.learning(part_num, iter_num, omega_max, omega_min, c1, c2)
        b_df = pd.DataFrame(b_list)
        MI_df = pd.DataFrame(MI_list)
        b_df.to_csv(output_path + 'region1_b_list_cv_{:d}.csv'.format(idx), header=False,
            index=False
            )
        MI_df.to_csv(output_path + 'region1_MI_list_cv_{:d}.csv'.format(idx), header=False,
            index=False
            )

        # region 2
        x_train = x2[train_idx]
        x_test = x2[test_idx]

        model = MI_learning(x_train, y_train, g2, k)
        b_list, MI_list = model.learning(part_num, iter_num, omega_max, omega_min, c1, c2)
        b_df = pd.DataFrame(b_list)
        MI_df = pd.DataFrame(MI_list)
        b_df.to_csv(output_path + 'region2_b_list_cv_{:d}.csv'.format(idx), header=False,
            index=False
            )
        MI_df.to_csv(output_path + 'region2_MI_list_cv_{:d}.csv'.format(idx), header=False,
            index=False
            )

        # region 3
        x_train = x3[train_idx]
        x_test = x3[test_idx]

        model = MI_learning(x_train, y_train, g3, k)
        b_list, MI_list = model.learning(part_num, iter_num, omega_max, omega_min, c1, c2)
        b_df = pd.DataFrame(b_list)
        MI_df = pd.DataFrame(MI_list)
        b_df.to_csv(output_path + 'region3_b_list_cv_{:d}.csv'.format(idx), header=False,
            index=False
            )
        MI_df.to_csv(output_path + 'region3_MI_list_cv_{:d}.csv'.format(idx), header=False,
            index=False
            )

        # region 4
        x_train = x4[train_idx]
        x_test = x4[test_idx]

        model = MI_learning(x_train, y_train, g4, k)
        b_list, MI_list = model.learning(part_num, iter_num, omega_max, omega_min, c1, c2)
        b_df = pd.DataFrame(b_list)
        MI_df = pd.DataFrame(MI_list)
        b_df.to_csv(output_path + 'region4_b_list_cv_{:d}.csv'.format(idx), header=False,
            index=False
            )
        MI_df.to_csv(output_path + 'region4_MI_list_cv_{:d}.csv'.format(idx), header=False,
            index=False
            )

        # region 5
        x_train = x5[train_idx]
        x_test = x5[test_idx]

        model = MI_learning(x_train, y_train, g5, k)
        b_list, MI_list = model.learning(part_num, iter_num, omega_max, omega_min, c1, c2)
        b_df = pd.DataFrame(b_list)
        MI_df = pd.DataFrame(MI_list)
        b_df.to_csv(output_path + 'region5_b_list_cv_{:d}.csv'.format(idx), header=False,
            index=False
            )
        MI_df.to_csv(output_path + 'region5_MI_list_cv_{:d}.csv'.format(idx), header=False,
            index=False
            )

    endtime = time.time()
    runtime = endtime - starttime
    print(runtime)


if __name__ == "__main__":
    main()