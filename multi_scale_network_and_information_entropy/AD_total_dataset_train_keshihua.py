import os
import glob
import argparse
import torch
import seaborn as sns
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from ggmfit import *
from mutual_information import *
from data_loader import *
from PSO import *
from MI_learning import *
from paper_network import *




def plot_embedding(X, Y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1], marker=('^' if Y[i] else 'o'),
        edgecolors=plt.cm.Set1(Y[i]),
        color='')

    plt.xticks([]), plt.yticks([])
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    if title is not None:
        plt.title(title)


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

    DATA_PATH = "D:/code/DTI_data/ADNI3_ADvsMCI_FN/"
    output_path = "D:/ASUS/code/mutual_information_ADNI3_output_total_dataset_train/AD_vs_MCI/"
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
    k = 3
    fs_num = 20
    pca_num = 20


    b_df = pd.read_csv(output_path + 'wholegraph_b_list_sparserate_{:.1f}.csv'.format(sparse_rate), header=None)
    b_list = b_df.to_numpy()
    b_list = b_list[:k]

    # machine learning
    f = []
    for b in b_list:
        f.append(np.matmul(x, b))
    f = np.array(f)
    f = np.transpose(f, (1, 0, 2))
    f = np.reshape(f, (f.shape[0], -1))

    # # fisher scoring
    # class0 = f[y == 0]
    # class1 = f[y == 1]
    # Mu0 = np.mean(class0, axis=0)
    # Mu1 = np.mean(class1, axis=0)
    # Mu = np.mean(f, axis=0)
    # Sigma0 = np.var(class0, axis=0)
    # Sigma1 = np.var(class1, axis=0)
    # n0 = class0.shape[0]
    # n1 = class1.shape[0]
    # fisher_score = (n0 * (Mu0 - Mu)**2 + n1 * (Mu1 - Mu)**2) / (n0 * Sigma0 + n1 * Sigma1)
    # sort_idx = np.argsort(fisher_score)[::-1]

    # sort_idx = sort_idx[:fs_num]
    
    # f = f[:, sort_idx]

    # Norm
    scaler = StandardScaler()
    scaler.fit(f)
    fscale = scaler.transform(f)


    # Ramdom Forest
    rf = RandomForestClassifier(max_depth=5, random_state=0)
    model = rf.fit(fscale, y)

    predict = model.predict(fscale)
    correct = np.sum(predict == y)
    accuracy = correct / y.size

    print("acc.: {:.1f}\n".format(accuracy*100))

    # Draw
    X_draw = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(fscale)
    plot_embedding(X_draw, y)
    plt.show()


if __name__ == "__main__":
    main()