import numpy as np
import pandas as pd
import sklearn.metrics
import scipy.sparse

from . import coarsening


def dist_graph(PATH, theta, k):
    dataset = pd.read_csv(PATH, sep=" ", header=None)
    position = dataset.iloc[:, 0:3].to_numpy()
    dist = sklearn.metrics.pairwise_distances(position, metric="euclidean")
    w = dist_adjacency(dist, theta, k)
    return w, position


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
