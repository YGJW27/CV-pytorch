import numpy as np
import numpy.matlib

def ggmfit(S, G, maxIter):
    '''
    MLE for a precision matrix given known zeros in the graph
    S is empirical covariance matrix, numpy matrix
    G is graph structure, numpy matrix
    Hastie, Tibshirani & Friedman ("Elements" book, 2nd Ed, 2008, p633)
    '''
    convengenceFlag = False
    p = S.shape[0]
    W = S
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
            notzero = np.squeeze(np.asarray(notzero))
            S12_nz = S12[notzero]
            W11_nz = W11[notzero][:, notzero]

            beta = np.matlib.zeros((p-1, 1), dtype=W.dtype)
            beta[notzero] = W11_nz.I * S12_nz
            # W12 = W11 * beta
            W12 = W11 * beta
            W[notj][:, j] = W12
            W[j][:, notj] = W12.T

            if (i == (maxIter - 1)) or convengenceFlag:
                theta22 = 1 / (S22 - W12.T * beta)
                assert theta22 >= 0
                theta12 = - beta * theta22
                theta[j, j] = theta22
                theta[notj][:, j] = theta12
                theta[j][:, notj] = theta12.T

        if convengenceFlag:
            break

        normW_ = np.linalg.norm(W)
        if np.abs(normW_ - normW) < 10e-5:
            convengenceFlag = True
    
    W = (W + W.T) / 2
    theta = (theta + theta.T) / 2

    return W, theta, i


def main():
    S = np.matrix('10,1,5,4; 1,10,2,6; 5,2,10,3; 4,6,3,10', dtype=float)
    G = np.matrix('0,1,0,1; 1,0,1,0; 0,1,0,1; 1,0,1,0', dtype=float)
    W, theta, i = ggmfit(S, G, 100)
    print(W,'\n', theta, '\n', i, '\n')

if __name__ == "__main__":
    main()