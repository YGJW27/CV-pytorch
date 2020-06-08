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

    DATA_PATH = "D:/code/DTI_data/Site-SI_FN/"
    output_path = "D:/code/mutual_information_sex_output/"
    filelist = data_list(DATA_PATH)
    dataset = MRI_Dataset(filelist)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    for data, target, idx in dataloader:
        x = data.numpy()
        y = target.numpy()
        idx = idx.numpy()

    # node_idx = nodes_selection()
    # x = x[:, node_idx, :][:, :, node_idx]
    x = np.tanh(x / 10)

    seed = 123456
    np.random.seed(seed)
    starttime = time.time()

    fs_num = 100

    # 10-fold validation
    acc_sum = 0
    cv = 10
    zero_one_mat_list = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    for idx, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # matrix to array
        shape = x_train.shape
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        x0 = x_train[y_train == 0]
        x1 = x_train[y_train == 1]
        t_value, p_value = scipy.stats.ttest_ind(x0, x1, axis=0, equal_var=False, nan_policy='omit')
        p_value[np.isnan(p_value)] = 1
        sort_idx = np.argsort(p_value)
        
        # find nodes with low p-value
        p_mat = p_value.reshape(shape[1], shape[2])
        zero_one_array = np.zeros((shape[1] * shape[2]))
        zero_one_array[sort_idx[:fs_num]] = 1
        zero_one_mat_list.append(zero_one_array.reshape(shape[1], shape[2]))


        # # classification
        # f_train = x_train[:, sort_idx[:fs_num]]
        # f_test = x_test[:, sort_idx[:fs_num]]

        # # Norm
        # scaler = StandardScaler()
        # scaler.fit(f_train)
        # fscale_train = scaler.transform(f_train)
        # fscale_test = scaler.transform(f_test)

        # # SVC
        # svc = SVC(kernel='rbf', random_state=1, gamma=1, C=1)
        # model = svc.fit(fscale_train, y_train)

        # predict_train = model.predict(fscale_train)
        # correct_train = np.sum(predict_train == y_train)
        # accuracy_train = correct_train / train_idx.size

        # predict = model.predict(fscale_test)
        # correct = np.sum(predict == y_test)
        # # print()
        # # print(classification_report(y_test, predict))
        # accuracy = correct / test_idx.size
        # print("cv: {}/{}, acc.: {:.1f}/{:.1f}\n".format(idx, cv, accuracy_train*100, accuracy*100))
        # acc_sum += accuracy
        # print("total acc.: {:.1f}\n".format(acc_sum / cv * 100))
        # print()

    consensus_mat = np.ones((shape[1], shape[2]))
    for mat in zero_one_mat_list:
        consensus_mat = consensus_mat * mat

    assert np.all(consensus_mat == consensus_mat.T)

    consensus_nodes = np.where(np.sum(consensus_mat, axis=0) != 0)[0]
    print(consensus_nodes)


    endtime = time.time()
    runtime = endtime - starttime
    print(runtime)


if __name__ == "__main__":
    main()