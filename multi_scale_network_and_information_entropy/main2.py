# main.py for machine learning
import os
import glob
import time
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
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
    DATA_PATH = "D:/code/DTI_data/network_FN/"
    NODE_PATH = "D:/code/DTI_data/network_distance/AAL_90_num.node"
    GRAPH_PATH = "D:/code/DTI_data/network_distance/grouplevel.edge"
    output_path = "D:/code/mutual_information_MDD_output/sparserate0.3_k4/"
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

    sparse_rate = 0.3
    g1 = graph_mine(x1, y, sparse_rate)
    g2 = graph_mine(x2, y, sparse_rate)
    g3 = graph_mine(x3, y, sparse_rate)
    g4 = graph_mine(x4, y, sparse_rate)
    g5 = graph_mine(x5, y, sparse_rate)

    seed = 123456
    np.random.seed(seed)
    starttime = time.time()

    fs_num = 3

    # 10-fold validation
    cv = 10
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    acc_sum1 = 0
    acc_sum2 = 0
    acc_sum3 = 0
    acc_sum4 = 0
    acc_sum5 = 0
    ensemble_acc_sum = 0
    for idx, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        y_train = y[train_idx]
        y_test = y[test_idx]

        #---------------------------- region 1 -----------------------------#
        x_train = x1[train_idx]
        x_test = x1[test_idx]

        # shape = x_train.shape
        # x_train = np.reshape(x_train, (x_train.shape[0], -1))
        # class0 = x_train[y_train == 0]
        # class1 = x_train[y_train == 1]
        # t_value, p_value = scipy.stats.ttest_ind(class0, class1, axis=0, equal_var=False, nan_policy='omit')
        # p_value[np.isnan(p_value)] = 1
        # sort_idx = np.argsort(p_value)
        # sort_idx = sort_idx[:fs_num * shape[1]]
        # f_train = x_train[:, sort_idx]
        # x_test = np.reshape(x_test, (x_test.shape[0], -1))
        # f_test = x_test[:, sort_idx]

        b_df = pd.read_csv(output_path + 'region1/region1_b_list_cv_{:d}.csv'.format(idx), header=None)
        b_list = b_df.to_numpy()

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
        sort_idx = sort_idx[:fs_num * x_train.shape[1]]
        
        f_train = f_train[:, sort_idx]
        f_test = f_test[:, sort_idx]

        # Norm
        scaler = StandardScaler()
        scaler.fit(f_train)
        fscale_train = scaler.transform(f_train)
        fscale_test = scaler.transform(f_test)

        # SVC
        svc1 = SVC(kernel='rbf', random_state=1, gamma=0.001, C=1)
        model = svc1.fit(fscale_train, y_train)

        predict1_train = model.predict(fscale_train)
        correct_train = np.sum(predict1_train == y_train)
        accuracy_train = correct_train / train_idx.size

        predict1 = model.predict(fscale_test)
        correct = np.sum(predict1 == y_test)
        # print()
        # print(classification_report(y_test, predict1))
        accuracy = correct / test_idx.size
        print("cv: {}/{}, acc.: {:.1f}/{:.1f}\n".format(idx, cv, accuracy_train*100, accuracy*100))
        acc_sum1 += accuracy
        print("total acc.: {:.1f}\n".format(acc_sum1 / cv * 100))
        print()


        # #---------------------------- region 2 -----------------------------#
        # x_train = x2[train_idx]
        # x_test = x2[test_idx]

        # # shape = x_train.shape
        # # x_train = np.reshape(x_train, (x_train.shape[0], -1))
        # # class0 = x_train[y_train == 0]
        # # class1 = x_train[y_train == 1]
        # # t_value, p_value = scipy.stats.ttest_ind(class0, class1, axis=0, equal_var=False, nan_policy='omit')
        # # p_value[np.isnan(p_value)] = 1
        # # sort_idx = np.argsort(p_value)
        # # sort_idx = sort_idx[:fs_num * shape[1]]
        # # f_train = x_train[:, sort_idx]
        # # x_test = np.reshape(x_test, (x_test.shape[0], -1))
        # # f_test = x_test[:, sort_idx]

        # b_df = pd.read_csv(output_path + 'region2/region2_b_list_cv_{:d}.csv'.format(idx), header=None)
        # b_list = b_df.to_numpy()

        # f_train = []
        # f_test = []
        # for b in b_list:
        #     f_train.append(np.matmul(x_train, b))
        #     f_test.append(np.matmul(x_test, b))
        # f_train = np.array(f_train)
        # f_train = np.transpose(f_train, (1, 0, 2))
        # f_train = np.reshape(f_train, (f_train.shape[0], -1))
        # f_test = np.array(f_test)
        # f_test = np.transpose(f_test, (1, 0, 2))
        # f_test = np.reshape(f_test, (f_test.shape[0], -1))

        # # fisher scoring
        # class0 = f_train[y_train == 0]
        # class1 = f_train[y_train == 1]
        # Mu0 = np.mean(class0, axis=0)
        # Mu1 = np.mean(class1, axis=0)
        # Mu = np.mean(f_train, axis=0)
        # Sigma0 = np.var(class0, axis=0)
        # Sigma1 = np.var(class1, axis=0)
        # n0 = class0.shape[0]
        # n1 = class1.shape[0]
        # fisher_score = (n0 * (Mu0 - Mu)**2 + n1 * (Mu1 - Mu)**2) / (n0 * Sigma0 + n1 * Sigma1)
        # sort_idx = np.argsort(fisher_score)[::-1]
        # sort_idx = sort_idx[:fs_num * x_train.shape[1]]
        
        # f_train = f_train[:, sort_idx]
        # f_test = f_test[:, sort_idx]

        # # Norm
        # scaler = StandardScaler()
        # scaler.fit(f_train)
        # fscale_train = scaler.transform(f_train)
        # fscale_test = scaler.transform(f_test)

        # # SVC
        # svc2 = SVC(kernel='rbf', random_state=1, gamma=1, C=1)
        # model = svc2.fit(fscale_train, y_train)

        # predict2_train = model.predict(fscale_train)
        # correct_train = np.sum(predict2_train == y_train)
        # accuracy_train = correct_train / train_idx.size

        # predict2 = model.predict(fscale_test)
        # correct = np.sum(predict2 == y_test)
        # # print()
        # # print(classification_report(y_test, predict2))
        # accuracy = correct / test_idx.size
        # print("cv: {}/{}, acc.: {:.1f}/{:.1f}\n".format(idx, cv, accuracy_train*100, accuracy*100))
        # acc_sum2 += accuracy
        # print("total acc.: {:.1f}\n".format(acc_sum2 / cv * 100))
        # print()


        # #---------------------------- region 3 -----------------------------#
        # x_train = x3[train_idx]
        # x_test = x3[test_idx]

        # # shape = x_train.shape
        # # x_train = np.reshape(x_train, (x_train.shape[0], -1))
        # # class0 = x_train[y_train == 0]
        # # class1 = x_train[y_train == 1]
        # # t_value, p_value = scipy.stats.ttest_ind(class0, class1, axis=0, equal_var=False, nan_policy='omit')
        # # p_value[np.isnan(p_value)] = 1
        # # sort_idx = np.argsort(p_value)
        # # sort_idx = sort_idx[:fs_num * shape[1]]
        # # f_train = x_train[:, sort_idx]
        # # x_test = np.reshape(x_test, (x_test.shape[0], -1))
        # # f_test = x_test[:, sort_idx]

        # b_df = pd.read_csv(output_path + 'region3/region3_b_list_cv_{:d}.csv'.format(idx), header=None)
        # b_list = b_df.to_numpy()

        # f_train = []
        # f_test = []
        # for b in b_list:
        #     f_train.append(np.matmul(x_train, b))
        #     f_test.append(np.matmul(x_test, b))
        # f_train = np.array(f_train)
        # f_train = np.transpose(f_train, (1, 0, 2))
        # f_train = np.reshape(f_train, (f_train.shape[0], -1))
        # f_test = np.array(f_test)
        # f_test = np.transpose(f_test, (1, 0, 2))
        # f_test = np.reshape(f_test, (f_test.shape[0], -1))

        # # fisher scoring
        # class0 = f_train[y_train == 0]
        # class1 = f_train[y_train == 1]
        # Mu0 = np.mean(class0, axis=0)
        # Mu1 = np.mean(class1, axis=0)
        # Mu = np.mean(f_train, axis=0)
        # Sigma0 = np.var(class0, axis=0)
        # Sigma1 = np.var(class1, axis=0)
        # n0 = class0.shape[0]
        # n1 = class1.shape[0]
        # fisher_score = (n0 * (Mu0 - Mu)**2 + n1 * (Mu1 - Mu)**2) / (n0 * Sigma0 + n1 * Sigma1)
        # sort_idx = np.argsort(fisher_score)[::-1]
        # sort_idx = sort_idx[:fs_num * x_train.shape[1]]
        
        # f_train = f_train[:, sort_idx]
        # f_test = f_test[:, sort_idx]

        # # Norm
        # scaler = StandardScaler()
        # scaler.fit(f_train)
        # fscale_train = scaler.transform(f_train)
        # fscale_test = scaler.transform(f_test)

        # # SVC
        # svc3 = SVC(kernel='rbf', random_state=1, gamma=0.001, C=100)
        # model = svc3.fit(fscale_train, y_train)

        # predict3_train = model.predict(fscale_train)
        # correct_train = np.sum(predict3_train == y_train)
        # accuracy_train = correct_train / train_idx.size

        # predict3 = model.predict(fscale_test)
        # correct = np.sum(predict3 == y_test)
        # # print()
        # # print(classification_report(y_test, predict3))
        # accuracy = correct / test_idx.size
        # print("cv: {}/{}, acc.: {:.1f}/{:.1f}\n".format(idx, cv, accuracy_train*100, accuracy*100))
        # acc_sum3 += accuracy
        # print("total acc.: {:.1f}\n".format(acc_sum3 / cv * 100))
        # print()


        # #---------------------------- region 4 -----------------------------#
        # x_train = x4[train_idx]
        # x_test = x4[test_idx]

        # # shape = x_train.shape
        # # x_train = np.reshape(x_train, (x_train.shape[0], -1))
        # # class0 = x_train[y_train == 0]
        # # class1 = x_train[y_train == 1]
        # # t_value, p_value = scipy.stats.ttest_ind(class0, class1, axis=0, equal_var=False, nan_policy='omit')
        # # p_value[np.isnan(p_value)] = 1
        # # sort_idx = np.argsort(p_value)
        # # sort_idx = sort_idx[:fs_num * shape[1]]
        # # f_train = x_train[:, sort_idx]
        # # x_test = np.reshape(x_test, (x_test.shape[0], -1))
        # # f_test = x_test[:, sort_idx]

        # b_df = pd.read_csv(output_path + 'region4/region4_b_list_cv_{:d}.csv'.format(idx), header=None)
        # b_list = b_df.to_numpy()

        # f_train = []
        # f_test = []
        # for b in b_list:
        #     f_train.append(np.matmul(x_train, b))
        #     f_test.append(np.matmul(x_test, b))
        # f_train = np.array(f_train)
        # f_train = np.transpose(f_train, (1, 0, 2))
        # f_train = np.reshape(f_train, (f_train.shape[0], -1))
        # f_test = np.array(f_test)
        # f_test = np.transpose(f_test, (1, 0, 2))
        # f_test = np.reshape(f_test, (f_test.shape[0], -1))

        # # fisher scoring
        # class0 = f_train[y_train == 0]
        # class1 = f_train[y_train == 1]
        # Mu0 = np.mean(class0, axis=0)
        # Mu1 = np.mean(class1, axis=0)
        # Mu = np.mean(f_train, axis=0)
        # Sigma0 = np.var(class0, axis=0)
        # Sigma1 = np.var(class1, axis=0)
        # n0 = class0.shape[0]
        # n1 = class1.shape[0]
        # fisher_score = (n0 * (Mu0 - Mu)**2 + n1 * (Mu1 - Mu)**2) / (n0 * Sigma0 + n1 * Sigma1)
        # sort_idx = np.argsort(fisher_score)[::-1]
        # sort_idx = sort_idx[:fs_num * x_train.shape[1]]
        
        # f_train = f_train[:, sort_idx]
        # f_test = f_test[:, sort_idx]

        # # Norm
        # scaler = StandardScaler()
        # scaler.fit(f_train)
        # fscale_train = scaler.transform(f_train)
        # fscale_test = scaler.transform(f_test)

        # # SVC
        # svc4 = SVC(kernel='rbf', random_state=1, gamma=0.001, C=100)
        # model = svc4.fit(fscale_train, y_train)

        # predict4_train = model.predict(fscale_train)
        # correct_train = np.sum(predict4_train == y_train)
        # accuracy_train = correct_train / train_idx.size

        # predict4 = model.predict(fscale_test)
        # correct = np.sum(predict4 == y_test)
        # # print()
        # # print(classification_report(y_test, predict4))
        # accuracy = correct / test_idx.size
        # print("cv: {}/{}, acc.: {:.1f}/{:.1f}\n".format(idx, cv, accuracy_train*100, accuracy*100))
        # acc_sum4 += accuracy
        # print("total acc.: {:.1f}\n".format(acc_sum4 / cv * 100))
        # print()


        # #---------------------------- region 5 -----------------------------#
        # x_train = x5[train_idx]
        # x_test = x5[test_idx]

        # # shape = x_train.shape
        # # x_train = np.reshape(x_train, (x_train.shape[0], -1))
        # # class0 = x_train[y_train == 0]
        # # class1 = x_train[y_train == 1]
        # # t_value, p_value = scipy.stats.ttest_ind(class0, class1, axis=0, equal_var=False, nan_policy='omit')
        # # p_value[np.isnan(p_value)] = 1
        # # sort_idx = np.argsort(p_value)
        # # sort_idx = sort_idx[:fs_num * shape[1]]
        # # f_train = x_train[:, sort_idx]
        # # x_test = np.reshape(x_test, (x_test.shape[0], -1))
        # # f_test = x_test[:, sort_idx]

        # b_df = pd.read_csv(output_path + 'region5/region5_b_list_cv_{:d}.csv'.format(idx), header=None)
        # b_list = b_df.to_numpy()

        # f_train = []
        # f_test = []
        # for b in b_list:
        #     f_train.append(np.matmul(x_train, b))
        #     f_test.append(np.matmul(x_test, b))
        # f_train = np.array(f_train)
        # f_train = np.transpose(f_train, (1, 0, 2))
        # f_train = np.reshape(f_train, (f_train.shape[0], -1))
        # f_test = np.array(f_test)
        # f_test = np.transpose(f_test, (1, 0, 2))
        # f_test = np.reshape(f_test, (f_test.shape[0], -1))

        # # fisher scoring
        # class0 = f_train[y_train == 0]
        # class1 = f_train[y_train == 1]
        # Mu0 = np.mean(class0, axis=0)
        # Mu1 = np.mean(class1, axis=0)
        # Mu = np.mean(f_train, axis=0)
        # Sigma0 = np.var(class0, axis=0)
        # Sigma1 = np.var(class1, axis=0)
        # n0 = class0.shape[0]
        # n1 = class1.shape[0]
        # fisher_score = (n0 * (Mu0 - Mu)**2 + n1 * (Mu1 - Mu)**2) / (n0 * Sigma0 + n1 * Sigma1)
        # sort_idx = np.argsort(fisher_score)[::-1]
        # sort_idx = sort_idx[:fs_num * x_train.shape[1]]
        
        # f_train = f_train[:, sort_idx]
        # f_test = f_test[:, sort_idx]

        # # Norm
        # scaler = StandardScaler()
        # scaler.fit(f_train)
        # fscale_train = scaler.transform(f_train)
        # fscale_test = scaler.transform(f_test)

        # # SVC
        # # svc5 = SVC(kernel='rbf', random_state=1, gamma=0.001, C=10) # shape * 2
        # svc5 = SVC(kernel='rbf', random_state=1, gamma=0.001, C=10) # shape * 1
        # model = svc5.fit(fscale_train, y_train)

        # predict5_train = model.predict(fscale_train)
        # correct_train = np.sum(predict5_train == y_train)
        # accuracy_train = correct_train / train_idx.size

        # predict5 = model.predict(fscale_test)
        # correct = np.sum(predict5 == y_test)
        # # print()
        # # print(classification_report(y_test, predict5))
        # accuracy = correct / test_idx.size
        # print("cv: {}/{}, acc.: {:.1f}/{:.1f}\n".format(idx, cv, accuracy_train*100, accuracy*100))
        # acc_sum5 += accuracy
        # print("total acc.: {:.1f}\n".format(acc_sum5 / cv * 100))
        # print()

        # #------------------------ Ensemble Learning ------------------------#
        # predicts_train = np.stack((predict1_train, predict2_train, predict3_train, predict4_train, predict5_train))
        # predicts_train = predicts_train.T
        # predicts = np.stack((predict1, predict2, predict3, predict4, predict5))
        # predicts = predicts.T
        # svc_ens = SVC(kernel='linear', C=1)
        # model = svc_ens.fit(predicts_train, y_train)

        # predicts_train = model.predict(predicts_train)
        # correct_train = np.sum(predicts_train == y_train)
        # accuracy_train = correct_train / train_idx.size

        # predicts = model.predict(predicts)
        # correct = np.sum(predicts == y_test)
        # print()
        # print(classification_report(y_test, predicts))
        # accuracy = correct / test_idx.size
        # print("cv: {}/{}, acc.: {:.1f}/{:.1f}\n".format(idx, cv, accuracy_train*100, accuracy*100))
        # ensemble_acc_sum += accuracy
        # print("total acc.: {:.1f}\n".format(ensemble_acc_sum / cv * 100))
        # print()


    endtime = time.time()
    runtime = endtime - starttime
    print(runtime)


if __name__ == "__main__":
    main()