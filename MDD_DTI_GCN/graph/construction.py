import numpy as np
import pandas as pd
import sklearn.metrics


def dist_graph(PATH, theta, k):
    dataset = pd.read_csv(PATH, sep=" ", header=None)
    position = dataset.iloc[:, 0:3].to_numpy()
    dist = sklearn.metrics.pairwise_distances(position, metric="euclidean")
    w = dist_adjacency(dist, theta, k)
    return w


def dist_adjacency(dist, theta, k):
    np.fill_diagonal(dist, k+1)
    mask = dist <= k
    w = np.zeros_like(dist)
    w[mask] = np.exp(-dist[mask]**2 / (2 * theta**2))
    return w


def to_graph_signal(w):
    pass
