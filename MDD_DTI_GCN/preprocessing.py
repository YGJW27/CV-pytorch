import numpy as np
import pandas as pd
import sklearn.metrics
import scipy.sparse

import coarsening


def dist_graph(PATH, delta):
    dataset = pd.read_csv(PATH, sep=" ", header=None)
    position = dataset.iloc[:, 0:3].to_numpy()
    dist = sklearn.metrics.pairwise_distances(position, metric="euclidean")
    w = np.zeros_like(dist)
    w = np.exp()
    return w


def dist_adjacency(dist, theta, k):
    np.fill_diagonal(dist, k+1)
    mask = dist <= k
    w = np.zeros_like(dist)
    w[mask] = np.exp(-dist[mask]**2 / (2 * theta**2))
    w = scipy.sparse.csr_matrix(w)
    return w


def to_graph_signal(w):
    M, M = w.shape
    w = scipy.sparse.csr_matrix(w)
    L = coarsening.laplacian(w, normalized=True)
    L = coarsening.rescaled_L(L).toarray()
    stimulus = np.expand_dims(np.ones(M, dtype=w.dtype), axis=1)
    response1 = np.dot(L, stimulus)
    response2 = 2 * np.dot(L, response1) - stimulus
    response = np.concatenate((response1.T, response2.T), axis=0)
    return response


def main():
    NODE_PATH = "D:/code/DTI_data/network_distance/AAL_116.node"
    delta = 30
    dist_A = dist_graph(NODE_PATH, delta)


if __name__ == "__main__":
    main()
