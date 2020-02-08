import os
import glob
import pandas as pd
import numpy as np
import torch


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


def graph_model(Ws):
    """
    Ws is a batch of adjancency matrices from n samples,
    whose size is (Sample_Num, Vertex_Num, Vertex_Num)
    """
    


def main():
    DATA_PATH = "D:/code/DTI_data/network_FN/"
    NODE_PATH = "D:/code/DTI_data/network_distance/AAL_90_num.node"
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
    
    graph_model(x1)






if __name__ == "__main__":
    main()