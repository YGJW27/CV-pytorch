import numpy as np
import networkx as nx

def graph_generate(shape, sample_num, random_seed=0):
    np.random.seed(0)
    x0_mu = np.random.randint(6, 15, size=shape)
    x0_cov = np.random.uniform(0, 1, size=(shape, shape))
    x0_cov = (x0_cov + x0_cov.T) / 2
    np.fill_diagonal(x0_cov, np.random.randint(1, 4, size=shape))
    x1_mu = np.random.randint(7, 13, size=shape)
    x1_cov = np.random.uniform(0, 1.2, size=(shape, shape))
    x1_cov = (x1_cov + x1_cov.T) / 2
    np.fill_diagonal(x1_cov, np.random.randint(2, 4, size=shape))

    np.random.seed(random_seed)

    x0 = np.random.multivariate_normal(x0_mu, x0_cov, shape * sample_num)
    x1 = np.random.multivariate_normal(x1_mu, x1_cov, shape * sample_num)
    y0 = np.zeros(sample_num)
    y1 = np.ones(sample_num)
    x = np.concatenate((x0, x1), axis=0)
    x = np.reshape(x, (-1, shape, shape))
    y = np.concatenate((y0, y1))
    assert x.shape[0] == sample_num * 2
    assert y.shape[0] == sample_num * 2

    w = np.zeros(shape=x.shape)
    for i, wi in enumerate(w):
        w[i] = np.rint((x[i] + x[i].T) / 2)
        np.fill_diagonal(w[i], 0)


    return w, y, (x0_mu, x0_cov, x1_mu, x1_cov)


def nxnetwork_generate(shape, sample_num, random_seed=0):
    G = nx.watts_strogatz_graph(shape, shape-1, 0.3, seed=random_seed)
    G0 = nx.watts_strogatz_graph(shape, shape-1, 0.3, seed=random_seed+1214)
    G00 = nx.watts_strogatz_graph(shape, shape-1, 0.3, seed=random_seed+121124)
    G000 = nx.watts_strogatz_graph(shape, shape-1, 0.3, seed=random_seed+66324)
    G0000 = nx.watts_strogatz_graph(shape, shape-1, 0.3, seed=random_seed+2124) # shape10
    G00000 = nx.watts_strogatz_graph(shape, shape-1, 0.3, seed=random_seed+345324) # shape10
    G1 = nx.watts_strogatz_graph(shape, shape-1, 0.3, seed=random_seed+5212)
    G11 = nx.watts_strogatz_graph(shape, shape-1, 0.3, seed=random_seed+521432)
    G111 = nx.watts_strogatz_graph(shape, shape-1, 0.3, seed=random_seed+98332)
    G1111 = nx.watts_strogatz_graph(shape, shape-1, 0.3, seed=random_seed+56634) # shape10
    G11111 = nx.watts_strogatz_graph(shape, shape-1, 0.3, seed=random_seed+13523) # shape10
    adj = nx.to_numpy_array(G)
    adj0 = nx.to_numpy_array(G0)
    adj00 = nx.to_numpy_array(G00)
    adj000 = nx.to_numpy_array(G000)
    adj0000 = nx.to_numpy_array(G0000)
    adj00000 = nx.to_numpy_array(G00000)
    adj1 = nx.to_numpy_array(G1)
    adj11 = nx.to_numpy_array(G11)
    adj111 = nx.to_numpy_array(G111)
    adj1111 = nx.to_numpy_array(G1111)
    adj11111 = nx.to_numpy_array(G11111)

    w_0 = np.zeros(shape=(sample_num, shape, shape))
    for i, wi in enumerate(w_0):
        weights_0= np.random.normal(10, 3, size=(shape, shape))
        weights_0[weights_0 < 1] = 1
        w_0[i] = np.rint((weights_0 + weights_0.T) / 2) * adj

        weights_0= np.random.normal(1, 1, size=(shape, shape))
        error = np.random.poisson(5, size=(shape, shape))
        w_0[i] = np.rint((weights_0 + weights_0.T) / 2) * adj0 + w_0[i] \
             + np.rint((error + error.T))

        weights_0 = np.random.normal(5, 1, size=(shape, shape))
        w_0[i] = np.rint((weights_0 + weights_0.T) / 2) * adj00 + w_0[i]

        weights_0 = np.random.normal(3, 0.6, size=(shape, shape))
        w_0[i] = np.rint((weights_0 + weights_0.T) / 2) * adj000 + w_0[i]

        # errorx = np.random.normal(3, 2, size=(shape, shape))    # shape10
        # w_0[i] = w_0[i] + np.rint((errorx + errorx.T))          # shape10
        # errorxx = np.random.uniform(0, 8, size=(shape, shape))  # shape10
        # w_0[i] = w_0[i] + np.rint((errorxx + errorxx.T))        # shape10

        np.fill_diagonal(w_0[i], 0)

    w_1 = np.zeros(shape=(sample_num, shape, shape))
    for i, wi in enumerate(w_1):
        weights_1 = np.random.normal(10, 3, size=(shape, shape))
        weights_1[weights_1 < 1] = 1
        w_1[i] = np.rint((weights_1 + weights_1.T) / 2) * adj

        weights_1 = np.random.normal(1, 0.5, size=(shape, shape))
        error = np.random.poisson(5, size=(shape, shape))
        w_1[i] = np.rint((weights_1 + weights_1.T) / 2) * adj1 + w_1[i] \
             + np.rint((error + error.T))

        weights_1 = np.random.normal(4.5, 1, size=(shape, shape))
        w_1[i] = np.rint((weights_1 + weights_1.T) / 2) * adj11 + w_1[i]

        weights_1 = np.random.normal(3, 0.2, size=(shape, shape))
        w_1[i] = np.rint((weights_1 + weights_1.T) / 2) * adj111 + w_1[i]

        # errorx = np.random.normal(3, 2, size=(shape, shape))    # shape10
        # w_1[i] = w_1[i] + np.rint((errorx + errorx.T))          # shape10
        # errorxx = np.random.uniform(0, 8, size=(shape, shape))  # shape10
        # w_1[i] = w_1[i] + np.rint((errorxx + errorxx.T))        # shape10

        np.fill_diagonal(w_1[i], 0)

    w = np.concatenate((w_0, w_1))
    y0 = np.zeros(sample_num)
    y1 = np.ones(sample_num)
    y = np.concatenate((y0, y1))

    return w, y, (adj, adj0, adj1)


if __name__ == "__main__":
    # w, y, _ = graph_generate(4, 100, random_seed=100) shape, sample_num, random_seed=0
    nxnetwork_generate(4, 100, 123)
    print()