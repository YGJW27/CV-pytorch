import numpy as np
import numpy.matlib

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
                theta[notj][:, j] = theta12
                theta[j][:, notj] = theta12.T

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