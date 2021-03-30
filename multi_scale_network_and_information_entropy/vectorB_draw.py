# main.py for MI learning
import os
import glob
import time
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

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
    parser.add_argument('-R', '--sparserate', type=float, default=0.3, metavar='S')
    parser.add_argument('-M', '--learnmethod', type=str, default="mi")
    args = parser.parse_args()

    DATA_PATH = "D:/Project/ADNI_data/dataset/ADNI3_ADvsMCI_FN/"
    output_path = "D:/Project/ADNI_data/mutual_information_ADNI3_output/AD_vs_MCI/"
    Node_path = "D:/ASUS/BrainNet_draw/Node_AAL90.node"
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

    sparse_rate = args.sparserate

    seed = 123456
    np.random.seed(seed)
    starttime = time.time()

    # MI learning
    fs_num = 30
    pca_num = 5


    cv = 10
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    for idx, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        if idx != 9:
            continue
        b_df = pd.read_csv(output_path + 'wholegraph_b_list_sparserate_{:.1f}_cv_{:d}.csv'.format(sparse_rate, idx), header=None)
        b_list = b_df.to_numpy()

        node_df = pd.read_csv(Node_path, sep='\t', header=None)
        print("end")

        for idx, b in enumerate(b_list):
            node_df.iloc[:, 3] = -10
            node_df.iloc[:, 3][node_idx] = b
            node_df.to_csv('D:/AD_MCI_b{:d}.node'.format(idx), sep='\t', header=False, index=False)


if __name__ == "__main__":
    main()