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

    node_idx = nodes_selection()
    x = x[:, node_idx, :][:, :, node_idx]
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

    # PSO parameters
    part_num = 30
    iter_num = 2000
    omega_max = 0.9
    omega_min = 0.4
    c1 = 2
    c2 = 2

    # 10-fold validation
    acc_sum = 0
    cv = 10
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    for idx, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        if not idx == args.idx:
            continue
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = MI_learning(x_train, y_train, g, k)
        b_list, MI_list, _ = model.learning(part_num, iter_num, omega_max, omega_min, c1, c2)
        b_df = pd.DataFrame(b_list)
        MI_df = pd.DataFrame(MI_list)
        b_df.to_csv(output_path + 'wholegraph_b_list_sparserate_{:.1f}_cv_{:d}.csv'.format(sparse_rate, idx), header=False,
            index=False
            )
        MI_df.to_csv(output_path + 'wholegraph_MI_list_sparserate_{:.1f}_cv_{:d}.csv'.format(sparse_rate, idx), header=False,
            index=False
            )

        # machine learning
        f_train = []
        f_test = []
        for b in b_list:
            f_train.append(np.matmul(x_train, b))
            f_test.append(np.matmul(x_test, b))
        f_train = np.array(f_train)
        f_train = np.transpose(f_train, (1, 0, 2))
        f_train = np.reshape(f_train, (f_train.shape[0], -1))
        f_test = np.array(f_test)
        f_test = np.transpose(f_test, (1, 0, 2))
        f_test = np.reshape(f_test, (f_test.shape[0], -1))

        # fisher scoring
        class0 = f_train[y_train == 0]
        class1 = f_train[y_train == 1]
        Mu0 = np.mean(class0, axis=0)
        Mu1 = np.mean(class1, axis=0)
        Mu = np.mean(f_train, axis=0)
        Sigma0 = np.var(class0, axis=0)
        Sigma1 = np.var(class1, axis=0)
        n0 = class0.shape[0]
        n1 = class1.shape[0]
        fisher_score = (n0 * (Mu0 - Mu)**2 + n1 * (Mu1 - Mu)**2) / (n0 * Sigma0 + n1 * Sigma1)
        sort_idx = np.argsort(fisher_score)[::-1]
        sort_idx = sort_idx[:fs_num]
        
        f_train = f_train[:, sort_idx]
        f_test = f_test[:, sort_idx]

        # Norm
        scaler = StandardScaler()
        scaler.fit(f_train)
        fscale_train = scaler.transform(f_train)
        fscale_test = scaler.transform(f_test)

        # SVC
        svc = SVC(kernel='rbf', random_state=1, gamma=1, C=1)
        model = svc.fit(fscale_train, y_train)

        predict_train = model.predict(fscale_train)
        correct_train = np.sum(predict_train == y_train)
        accuracy_train = correct_train / train_idx.size

        predict = model.predict(fscale_test)
        correct = np.sum(predict == y_test)
        # print()
        # print(classification_report(y_test, predict))
        accuracy = correct / test_idx.size
        print("cv: {}/{}, acc.: {:.1f}/{:.1f}\n".format(idx, cv, accuracy_train*100, accuracy*100))
        acc_sum += accuracy
        print("total acc.: {:.1f}\n".format(acc_sum / cv * 100))
        print()




    endtime = time.time()
    runtime = endtime - starttime
    print(runtime)


if __name__ == "__main__":
    main()