import os
import glob
import pandas as pd
import networkx as nx
import numpy as np
import torch
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, KBinsDiscretizer
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "D:/code/DTI_data/network_FN/"

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

    
filelist = data_list(DATA_PATH)
dataset = MRI_Dataset(filelist)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)

for data, target, idx in dataloader:
    x = data.numpy()
    y = target.numpy()
    idx = idx.numpy()
    
x = x.reshape(x.shape[0], -1)
# x = x[:, np.any(x, axis=0)]

def Shannon_entropy(A):
    unique, counts = np.unique(A, return_counts=True)
    p = counts/counts.sum()
    ent = -np.sum(p * np.log2(p))
    return ent

def mutual_information(A, B):
    H_A = Shannon_entropy(A)
    unique, counts= np.unique(B, return_counts=True)
    H_A1B = 0
    for idx, status in enumerate(unique):
        H_A1B += Shannon_entropy(A[B==status]) * counts[idx]/counts.sum()
    MI_AB = H_A - H_A1B
    return MI_AB

est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
est.fit(x)
x_d = est.transform(x)

MI_array = np.zeros(x_d.shape[1])
for i, e in enumerate(MI_array):
    MI_array[i] = Shannon_entropy(x_d[:, i])

MI_array = MI_array.reshape(data.shape[1], -1)
MI_array[MI_array>3] = 0
G_MI = nx.from_numpy_array(MI_array)

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

# seed = 1
# cv = 20
# kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
# acc_sum = 0
# for idx, (train_idx, test_idx) in enumerate(kf.split(dataset)):
#     x_t = x[train_idx]
#     y_t = y[train_idx]
    
#     est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
#     est.fit(x_t)
#     x_d = est.transform(x_t)

#     MI_array = np.zeros(x_d.shape[1])
#     for i, e in enumerate(MI_array):
#         MI_array[i] = Shannon_entropy(x_d[:, i])
    
#     x_d = x[:, MI_array>2.8]

#     ynew_train = y[train_idx]
#     ynew_test = y[test_idx]


#     # Norm
#     scaler = MaxAbsScaler()
#     scaler.fit(x_d[train_idx])
#     xnew_train = scaler.transform(x_d[train_idx])
#     xnew_test = scaler.transform(x_d[test_idx])
#     print(xnew_train.shape[1])
    

#     # SVC
#     svc = SVC(kernel='rbf', random_state=1, gamma=0.01, C=10)
#     model = svc.fit(xnew_train, ynew_train)

#     predict = model.predict(xnew_test)
#     correct = np.sum(predict == ynew_test)
#     accuracy = correct / test_idx.size
#     print("cv: {}/{}, acc.: {:.1f}\n".format(idx, cv, accuracy*100))
#     acc_sum += accuracy
# print("total acc.: {:.1f}\n".format(acc_sum / cv * 100))

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
node_path = 'D:/code/DTI_data/network_distance/AAL_90.node'
nodes = pd.read_csv(node_path, sep=' ', header=None)
nodes = nodes.iloc[:, 0:3]
avg = torch.mean(data, axis=0).numpy()
G = nx.from_numpy_array(avg)
pos = dict(zip(range(nodes.shape[0]), [list(row) for row in nodes.to_numpy()]))
nx.set_node_attributes(G, pos, 'coord')

nx.set_node_attributes(G_MI, pos, 'coord')


from mayavi import mlab

def draw_network(G):
    mlab.clf()
    pos = np.array([pos for key, pos in G.nodes('coord')])
    pts = mlab.points3d(pos[:, 0], pos[:, 1], pos[:, 2], resolution=20, scale_factor=5)
    pts.mlab_source.dataset.lines = np.array([row for row in G.edges(data='weight')])
    tube = mlab.pipeline.tube(pts,  tube_radius=0.5)
    mlab.pipeline.surface(tube)

    mlab.show()


draw_network(G_MI)
