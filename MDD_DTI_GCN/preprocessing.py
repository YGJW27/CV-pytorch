import os
import glob
import numpy as np
import pandas as pd
import sklearn.metrics
import scipy.sparse

import coarsening


def dist_graph(PATH, delta):
    dataset = pd.read_csv(PATH, sep=" ", header=None)
    position = dataset.iloc[:, 0:3].to_numpy()
    dist = sklearn.metrics.pairwise_distances(position, metric="euclidean")
    d = np.exp(-dist**2 / (2 * delta**2))
    return d


def know_graph(PATH):
    dataset = pd.read_csv(PATH, sep=" ", header=None)
    k = dataset.to_numpy()
    return k


def knn(w, k):
    idx = np.argsort(w)[:,::-1]
    idx = idx[:,1:k+1]
    value = np.sort(w)
    value = value[:,::-1]
    value = value[:,1:k+1]
    i = np.arange(0, w.shape[0]).repeat(k)
    j = idx.reshape(-1)
    data = value.reshape(-1)
    w_s = scipy.sparse.coo_matrix((data, (i, j)), shape=w.shape)

    bigger = w_s.T > w_s
    w_s = w_s - w_s.multiply(bigger) + w_s.T.multiply(bigger)
    return w_s


def sample_processing(sample_path, output_path):
    sub_dirs = [x[0] for x in os.walk(sample_path)]
    sub_dirs.pop(0)

    for sub_dir in sub_dirs:
        file_list = []
        dir_name = os.path.basename(sub_dir)
        file_glob = os.path.join(sample_path, dir_name, '*')
        file_list.extend(glob.glob(file_glob))
        output_dir = os.path.join(output_path, dir_name)
        if os.path.exists(output_dir):
            continue
        else:
            os.makedirs(output_dir)

        for f in file_list:
            file_name = os.path.basename(f)
            output = os.path.join(output_dir, file_name)
            dataframe = pd.read_csv(f, sep="\s+", header=None)
            w = dataframe.to_numpy()
            gs = to_graph_signal(w, ks=3)
            gs = pd.DataFrame(gs)
            gs.to_csv(output, sep="\t", header=False, index=False)


def to_graph_signal(w, ks):
    M, M = w.shape
    w = scipy.sparse.csr_matrix(w)
    L = coarsening.laplacian(w, normalized=True)
    L = coarsening.rescaled_L(L)
    L_list = []
    T_0 = scipy.sparse.identity(M, dtype=w.dtype, format='csr')
    T_1 = L
    L_list.append(T_1)
    if ks > 1:
        T_2 = 2 * L * T_1 - T_0
        L_list.append(T_2)
    for k in range(2, ks):
        T_k = 2 * L * L_list[k-1] - L_list[k-2]
        L_list.append(T_k)

    stimulus = stimulus_construction(w, "constant")     # list of arrays
    response = []
    for idx, x in enumerate(stimulus):
        res_i = []
        for l in L_list:
            res = np.sum(l.toarray()[idx] * x)
            res_i.append(res)
        response.append(res_i)

    return np.array(response)


def stimulus_construction(w, type):
    if type == "constant":
        return np.ones(w.shape, dtype=w.dtype)
    elif type == "linear":
        pass


def main():
    NODE_PATH = "D:/code/DTI_data/network_distance/AAL_90.node"
    KNOW_PATH = "D:/code/DTI_data/network_distance/know_graph.edge"
    DATA_PATH = "D:/code/DTI_data/network_FN/"
    OUTPUT_PATH = "D:/code/DTI_data/output/"
    delta = 30
    d = dist_graph(NODE_PATH, delta)
    k = know_graph(KNOW_PATH)
    w = k * d
    w_s = knn(w, 8)
    df = pd.DataFrame(w_s.toarray())
    df.to_csv("D:/code/DTI_data/network_distance/grouplevel.edge", sep="\t", header=False, index=False)
    sample_processing(DATA_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
