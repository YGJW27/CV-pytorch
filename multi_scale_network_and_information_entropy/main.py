import os
import glob
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
    output_path = "D:/code/mutual_information_MDD_output/"
    filelist = data_list(DATA_PATH)
    dataset = MRI_Dataset(filelist)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    for data, target, idx in dataloader:
        x = data.numpy()
        y = target.numpy()
        idx = idx.numpy()

    x = x / 10
    
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
    
    ggraph = pd.read_csv(GRAPH_PATH, sep='\t', header=None).to_numpy()
    g1 = np.matrix(ggraph[node1, :][:, node1])
    g2 = np.matrix(ggraph[node2, :][:, node2])
    g3 = np.matrix(ggraph[node3, :][:, node3])
    g4 = np.matrix(ggraph[node4, :][:, node4])
    g5 = np.matrix(ggraph[node5, :][:, node5])

    seed = 123456
    np.random.seed(seed)
    starttime = time.time()

    # PSO parameters
    part_num = 30
    iter_num = 5000
    omega_max = 0.9
    omega_min = 0.4
    c1 = 2
    c2 = 2

    # 10-fold validation
    cv = 10
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    acc_sum = 0
    for idx, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        if idx == 0:
            continue
        x1_train = x1[train_idx]
        x1_test = x1[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        
        model = MI_learning(x1_train, y_train, g1, 2)
        b_list, MI_list = model.learning(part_num, iter_num, omega_max, omega_min, c1, c2)
        b_df = pd.DataFrame(b_list)
        MI_df = pd.DataFrame(MI_list)
        b_df.to_csv(output_path + 'region1_b_list_cv_{:d}.csv'.format(idx), header=False,
            index=False
            )
        MI_df.to_csv(output_path + 'region1_MI_list_cv_{:d}.csv'.format(idx), header=False,
            index=False
            )

        # f1_train = np.matmul(x1_train, b_list[0])
        # f1_test = np.matmul(x1_test, b_list[0])
        # df = pd.DataFrame(np.array([part_num, iter_num, b1, MI_array1]).reshape(1, -1), columns=['part_num', 'iter_num', 'b', 'MI'])
        # plt.plot(MI_array1)
        # print(MI_array1[-1], '\n')

        # df.to_csv(output_path + \
        #     'x1_w1_part_num_{:d}.csv'.format(part_num),
        #     index=False
        #     )
        # plt.savefig(output_path + \
        #     'x1_w1_part_num_{:d}.png'.format(part_num),
        #     )

        # # Norm
        # scaler = StandardScaler()
        # scaler.fit(f1_train)
        # f1scale_train = scaler.transform(f1_train)
        # f1scale_test = scaler.transform(f1_test)

        # # PCA
        # # pca = PCA(n_components=70)
        # # pca.fit(xscale_train)
        # # xnew_train = (xscale_train)
        # # xnew_test = (xscale_test)


        # # SVC
        # svc = SVC(kernel='rbf', random_state=1, gamma=0.001, C=10)
        # model = svc.fit(f1scale_train, y_train)

        # predict_train = model.predict(f1scale_train)
        # correct_train = np.sum(predict_train == y_train)
        # accuracy_train = correct_train / train_idx.size

        # predict = model.predict(f1scale_test)
        # correct = np.sum(predict == y_test)
        # accuracy = correct / test_idx.size
        # print("cv: {}/{}, acc.: {:.1f}/{:.1f}\n".format(idx, cv, accuracy_train*100, accuracy*100))
        # acc_sum += accuracy
        # print("total acc.: {:.1f}\n".format(acc_sum / cv * 100))

    endtime = time.time()
    runtime = endtime - starttime
    print(runtime)


if __name__ == "__main__":
    main()