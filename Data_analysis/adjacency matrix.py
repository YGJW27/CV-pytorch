import os
import glob
import pandas as pd
import networkx as nx
import numpy as np
import torch
import scipy.stats

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


from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

seed = 1
cv = 20
kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
acc_sum = 0
for idx, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    x0 = x[train_idx][y[train_idx] == 0]
    x1 = x[train_idx][y[train_idx] == 1]
    # x0 = x[y == 0]
    # x1 = x[y == 1]
    t_value, p_value = scipy.stats.ttest_ind(x0, x1, axis=0, equal_var=False, nan_policy='omit')
    x_t = x[:, p_value <=0.03]
    print(np.where(p_value <=0.03))
    print(x_t.shape, y.shape)

    ynew_train = y[train_idx]
    ynew_test = y[test_idx]

    # Norm
    scaler = StandardScaler()
    scaler.fit(x_t[train_idx])
    xnew_train = scaler.transform(x_t[train_idx])
    xnew_test = scaler.transform(x_t[test_idx])

    # SVC
    svc = SVC(kernel='rbf', random_state=1, gamma=0.01, C=10)
    model = svc.fit(xnew_train, ynew_train)

    predict = model.predict(xnew_test)
    correct = np.sum(predict == ynew_test)
    accuracy = correct / test_idx.size
    print("cv: {}/{}, acc.: {:.1f}\n".format(idx, cv, accuracy*100))
    acc_sum += accuracy
print("total acc.: {:.1f}\n".format(acc_sum / cv * 100))
