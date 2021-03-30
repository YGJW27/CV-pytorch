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
from sklearn.ensemble import RandomForestClassifier

from ggmfit import *
from mutual_information import *
from PSO import *
from MI_learning import *
from paper_network import *


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
    parser.add_argument('-R', '--sparserate', type=float, default=0.2, metavar='S')
    args = parser.parse_args()

    DATA_PATH = "D:/yjw/code/DTI_data/ADNI3_ADvsMCI_FN/"
    output_path = "D:/yjw/code/mutual_information_ADNI3_output_total_dataset_train/AD_vs_MCI/"
    filelist = data_list(DATA_PATH)
    dataset = MRI_Dataset(filelist)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    for data, target, idx in dataloader:
        x = data.numpy()
        y = target.numpy()
        idx = idx.numpy()

    node_idx = nodes_selection_ADNI()
    x = x[:, node_idx, :][:, :, node_idx]

    x = noise_filter(x, 0.05)

    x = np.tanh(x / 10)

    # graph mine
    sparse_rate = args.sparserate
    g = graph_mine(x, y, sparse_rate)

    seed = 123456
    np.random.seed(seed)
    starttime = time.time()

    # MI learning
    k = 4
    fs_num = 20
    pca_num = 20

    # PSO parameters
    part_num = 30
    iter_num = 2000
    omega_max = 0.9
    omega_min = 0.4
    c1 = 2
    c2 = 2

    # MI learning

    model = MI_learning(x, y, g, k)
    b_list, MI_list, _ = model.learning(part_num, iter_num, omega_max, omega_min, c1, c2)
    b_df = pd.DataFrame(b_list)
    MI_df = pd.DataFrame(MI_list)
    b_df.to_csv(output_path + 'wholegraph_b_list_sparserate_{:.1f}.csv'.format(sparse_rate),
        header=False,
        index=False
        )


    endtime = time.time()
    runtime = endtime - starttime
    print(runtime)


if __name__ == "__main__":
    main()