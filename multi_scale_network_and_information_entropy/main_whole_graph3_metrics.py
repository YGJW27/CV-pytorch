import os
import glob
import time
import argparse
import pandas as pd
import numpy as np
import networkx as nx
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
    parser.add_argument('-R', '--sparserate', type=float, default=0.4, metavar='S')
    args = parser.parse_args()

    DATA_PATH = "D:/code/DTI_data/ADNI3_ADvsCN_FN/"
    filelist = data_list(DATA_PATH)
    dataset = MRI_Dataset(filelist)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    for data, target, idx in dataloader:
        x = data.numpy()
        y = target.numpy()
        idx = idx.numpy()

    node_idx = nodes_selection_ADNI()
    # x = x[:, node_idx, :][:, :, node_idx]

    x = noise_filter(x, 0.05)

    x = np.tanh(x / 10)

    seed = 123456
    np.random.seed(seed)
    starttime = time.time()

    # MI learning
    fs_num = 20

    # 10-fold validation
    acc_sum = 0
    TN_sum = 0
    TP_sum = 0
    cv = 10
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    for idx, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        f_train = []
        f_test = []
        for adj in x_train:
            G = nx.from_numpy_array(adj)
            cluster_coeff = nx.clustering(G)
            degree = G.degree(weight='weight')
            neighbor_degree = nx.average_neighbor_degree(G, weight='weight')
            coeff_array = np.zeros(x_train.shape[1])
            degree_array = np.zeros(x_train.shape[1])
            neighbor_degree_array = np.zeros(x_train.shape[1])
            for key, value in cluster_coeff.items():
                coeff_array[key] = value
            for key, value in degree:
                degree_array[key] = value
            for key, value in neighbor_degree.items():
                neighbor_degree_array[key] = value
            f_array = np.concatenate((coeff_array, degree_array, neighbor_degree_array))
            f_train.append(f_array)

        for adj in x_test:
            G = nx.from_numpy_array(adj)
            cluster_coeff = nx.clustering(G)
            degree = G.degree(weight='weight')
            neighbor_degree = nx.average_neighbor_degree(G, weight='weight')
            coeff_array = np.zeros(x_train.shape[1])
            degree_array = np.zeros(x_train.shape[1])
            neighbor_degree_array = np.zeros(x_train.shape[1])
            for key, value in cluster_coeff.items():
                coeff_array[key] = value
            for key, value in degree:
                degree_array[key] = value
            for key, value in neighbor_degree.items():
                neighbor_degree_array[key] = value
            f_array = np.concatenate((coeff_array, degree_array, neighbor_degree_array))
            f_test.append(f_array)

        

        f_train = np.array(f_train)
        f_test = np.array(f_test)
        

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


        # Ramdom Forest
        rf = RandomForestClassifier(max_depth=5, random_state=0)
        model = rf.fit(fscale_train, y_train)


        # SVC
        # svc = SVC(kernel='rbf', random_state=1, gamma=0.0001, C=1000)
        # model = svc.fit(fscale_train, y_train)

        predict_train = model.predict(fscale_train)
        correct_train = np.sum(predict_train == y_train)
        accuracy_train = correct_train / train_idx.size

        predict = model.predict(fscale_test)
        correct = np.sum(predict == y_test)

        TN = np.sum(predict[y_test == 0] == y_test[y_test == 0])
        TP = np.sum(predict[y_test == 1] == y_test[y_test == 1])
        TN_sum = TN_sum + TN
        TP_sum = TP_sum + TP
        # print()
        # print(classification_report(y_test, predict))
        accuracy = correct / test_idx.size
        print("cv: {}/{}, acc.: {:.1f}/{:.1f}\n".format(idx, cv, accuracy_train*100, accuracy*100))
        acc_sum += accuracy
        print("total acc.: {:.1f}\n".format(acc_sum / cv * 100))
        print()

    Acc = (TN_sum + TP_sum) / y.shape[0]
    Sen = TP_sum / np.sum(y)
    Spe = TN_sum / (y.shape[0] - np.sum(y))

    print("ACC: {:.2f}%, SEN: {:.2f}%, SPE: {:.2f}%".format(Acc * 100, Sen * 100, Spe * 100))




    endtime = time.time()
    runtime = endtime - starttime
    print(runtime)


if __name__ == "__main__":
    main()