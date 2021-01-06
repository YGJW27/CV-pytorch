import numpy as np
import numpy.matlib
import scipy.stats

def emp_covar_mat(dataset):
    '''
    calculate empirical covariance matrix S,
    dataset is given as 2-dimension numpy array (Sample_Num, Node_Num)
    '''
    SN = dataset.shape[0]
    NN = dataset.shape[1]
    S = np.matlib.zeros((NN, NN), dtype=dataset.dtype)
    x_aver = np.sum(dataset, axis=0) / SN
    x_aver_mat = np.matrix(x_aver)
    for x in dataset:
        x_mat = np.matrix(x)
        S += (x_mat - x_aver_mat).T * (x_mat - x_aver_mat)
    S = S / SN

    return S

def ggmfit(S, G, maxIter):
    '''
    MLE for a precision matrix given known zeros in the graph,
    S is empirical covariance matrix, numpy matrix,
    G is graph structure, numpy matrix,
    Hastie, Tibshirani & Friedman ("Elements" book, 2nd Ed, 2008, p634)
    '''
    S = np.matrix(S)
    G = np.matrix(G)

    convengenceFlag = False
    p = S.shape[0]
    W = np.copy(S)
    W = np.matrix(W)
    theta = np.matlib.zeros((p, p), dtype=W.dtype)  # precision matrix
    for i in range(maxIter):
        normW = np.linalg.norm(W)
        for j in range(p):
            notj = list(range(p))
            notj.pop(j)
            W11 = W[notj][:, notj]
            S12 = S[notj][:, j]
            S22 = S[j, j]

            # non-zero
            notzero = ~(G[j][:, notj] == 0)
            notzero = np.squeeze(np.asarray(notzero), axis=0)
            S12_nz = S12[notzero]
            W11_nz = W11[notzero][:, notzero]

            beta = np.matlib.zeros((p-1, 1), dtype=W.dtype)
            beta[notzero] = W11_nz.I * S12_nz
            # W12 = W11 * beta
            W12 = W11 * beta
            W[notj, j] = W12.T      # pay attention to this line.
            W[j, notj] = W12.T

            if (i == (maxIter - 1)) or convengenceFlag:
                theta22 = max(0, 1 / (S22 - W12.T * beta))
                theta12 = - beta * theta22
                theta[j, j] = theta22
                theta[notj, j] = theta12.T
                theta[j, notj] = theta12.T

        if convengenceFlag:
            break

        normW_ = np.linalg.norm(W)
        delta = np.abs(normW_ - normW)
        if delta < 10e-6:
            convengenceFlag = True
    
    W = (W + W.T) / 2
    theta = (theta + theta.T) / 2

    return W, theta, (i, delta)


def ggmfit_gradient(S, G, maxIter):
    S = np.array(S)
    G = np.array(G)
    np.fill_diagonal(G, 1)

    theta = np.linalg.inv(S) * (~(G == 0))
    for i in range(maxIter):
        grad = np.linalg.inv(theta) - S
        grad_constrain = grad * (~(G == 0))
        theta = theta + grad_constrain * 0.0005 * (1 + 0 * i / maxIter)
        delta = np.linalg.norm(grad_constrain)
        if i % 1000 == 0:
            print("{:.6f}\n".format(delta))
        if delta < 10e-5:
            break

    theta = (theta + theta.T) / 2
    return np.linalg.inv(theta), theta, (i, delta)


def graph_mine(x, y, sparse_rate):
    shape = x.shape
    if sparse_rate == 1:
        return np.ones((shape[1], shape[2]))

    x = x.reshape(shape[0], -1)
    x0 = x[y == 0]
    x1 = x[y == 1]
    t_value, p_value = scipy.stats.ttest_ind(x0, x1, axis=0, equal_var=False, nan_policy='omit')
    p_mat = p_value.reshape(shape[1], shape[2])
    p_mat[np.isnan(p_mat)] = 1
    assert np.all(p_mat == p_mat.T) == 1
    g = sparse_graph(p_mat, sparse_rate)

    return g


def sparse_graph(mat, sparse_rate):
    shape = mat.shape
    graph = np.zeros(shape)
    graph[np.arange(0, shape[0]), np.argmin(mat, axis=1)] = 1
    graph[np.argmin(mat, axis=1), np.arange(0, shape[0])] = 1

    if sparse_rate == 0:
        return graph

    mat = mat.reshape(-1)
    sort_idx = np.argsort(mat)
    threshold = mat[sort_idx[int(np.ceil((shape[0] * (shape[1] - 1)) * sparse_rate))]]
    graph[(mat <= threshold).reshape(shape[0], shape[1])] = 1
    assert np.all(graph == graph.T) == 1
    
    return graph


def noise_filter(w, thres_rate):
    w_non_zero = np.sum(w, axis=0) != 0
    w_thres = np.mean(w) * thres_rate
    w_mean = np.mean(w, axis=0)
    w_mask = np.ones(shape=w_mean.shape)
    w_mask[w_mean < w_thres] = 0
    w = w * w_mask
    w_non_zero_afterthres = np.sum(w, axis=0) != 0
    del_weights = (np.sum(w_non_zero) - np.sum(w_non_zero_afterthres)) / np.sum(w_non_zero)
    print("delete weights: ", del_weights)
    return w


def main():
    import pandas as pd

    S_df = pd.read_csv('D:/code/empirical.csv', header=None)
    G_df = pd.read_csv('D:/code/G.csv', header=None)
    S = S_df.to_numpy()
    G = G_df.to_numpy()
    W, theta, i = ggmfit(S, G, 100)
    print(W,'\n', theta, '\n', i, '\n')

if __name__ == "__main__":
    main()