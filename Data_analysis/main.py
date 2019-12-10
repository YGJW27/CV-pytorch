import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels as sm

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import scipy.sparse


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
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __getitem__(self, idx):
        filepath, target = self.data_list[idx][0], int(self.data_list[idx][1])
        dataframe = pd.read_csv(filepath, sep="\t", header=None)
        pic = dataframe.to_numpy()          # (V, C)
        pic = np.transpose(pic, (1, 0))     # (C, V)

        if self.transform is not None:
            pic = self.transform(pic)

        return pic, target, idx

    def __len__(self):
        return len(self.data_list)


class Array_To_Tensor(object):
    def __call__(self, input):
        return torch.from_numpy(input).float()


DATAPATH = 'D:/code/DTI_data/output/local_metrics_box/'
dataset = data_list(DATAPATH)

data_loader = torch.utils.data.DataLoader(
    MRI_Dataset(dataset,
                transform=transforms.Compose([
                    Array_To_Tensor()
                ])),
    batch_size=dataset.shape[0], shuffle=False)

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

seed = 4
cv = 20
kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
acc_sum = 0
for idx, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    for x, y, index in data_loader:
        x = x.numpy()
        y = y.numpy()
        x = x.reshape((-1, 270))
    ynew_train = y[train_idx]
    ynew_test = y[test_idx]

    # fisher score
    class0 = x[train_idx][ynew_train == 0]
    class1 = x[train_idx][ynew_train == 1]

    Mu0 = np.mean(class0, axis=0)
    Mu1 = np.mean(class1, axis=0)
    Mu = np.mean(x[train_idx], axis=0)
    Sigma0 = np.var(class0, axis=0)
    Sigma1 = np.var(class1, axis=0)
    n0 = class0.shape[0]
    n1 = class1.shape[0]
    fisher_score = (n0 * (Mu0 - Mu)**2 + n1 * (Mu1 - Mu)**2) / (n0 * Sigma0 + n1 * Sigma1)
    sort_idx = np.argsort(fisher_score)[::-1]

    # Norm
    scaler = StandardScaler()
    scaler.fit(x[train_idx])
    xscale_train = scaler.transform(x[train_idx])
    xscale_test = scaler.transform(x[test_idx])

    # PCA
    pca = PCA(n_components=70)
    pca.fit(xscale_train)
    xnew_train = (xscale_train)
    xnew_test = (xscale_test)

    xnew_train = xnew_train[:, sort_idx[0:40]]
    xnew_test = xnew_test[:, sort_idx[0:40]]

    # SVC
    svc = SVC(kernel='rbf', random_state=1, gamma=0.001, C=10)
    model = svc.fit(xnew_train, ynew_train)

    predict_train = model.predict(xnew_train)
    correct_train = np.sum(predict_train == ynew_train)
    accuracy_train = correct_train / train_idx.size

    predict = model.predict(xnew_test)
    correct = np.sum(predict == ynew_test)
    accuracy = correct / test_idx.size
    print("cv: {}/{}, acc.: {:.1f}\n".format(idx, cv, accuracy*100))
    acc_sum += accuracy
print("total acc.: {:.1f}\n".format(acc_sum / cv * 100))

