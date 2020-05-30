from ggmfit import *

def ggm_entropy(theta):
    NN = theta.shape[0]
    ent = NN / 2.0 * np.log2(2 * np.pi * np.e) - 1 / 2.0 * np.log2(max(np.linalg.det(theta), 10e-8))
    return ent


def mutual_information(F, Y, G):
    '''
    F is (Sample_Num, Node_Num), stack of feature vectors
    '''
    S = emp_covar_mat(F)
    W, theta, i = ggmfit(S, G, 5000)
    H_X = ggm_entropy(theta)

    unique, counts= np.unique(Y, return_counts=True)
    H_X1Y = 0
    for idx, status in enumerate(unique):
        S = emp_covar_mat(F[Y==status])
        W, theta, i = ggmfit(S, G, 5000)
        H_X1Y += ggm_entropy(theta) * counts[idx]/counts.sum()
    MI_XY = H_X - H_X1Y
    return MI_XY


def mutual_information_multinormal(F, Y):
    zero_num = np.sum(Y == 0)
    one_num = np.sum(Y == 1)
    n = F.shape[0]
    assert n == zero_num + one_num
    X = np.matrix(F).T
    J = np.matrix(np.ones((n, n)))
    I = np.matrix(np.identity(n))
    C = I - J / n
    S = X * C * X.T / n
    lam = n * np.log2(np.linalg.det(S))

    X0 = X * I[:, : zero_num]
    X1 = X * I[:, zero_num :]
    J0 = np.matrix(np.ones((zero_num, zero_num)))
    J1 = np.matrix(np.ones((one_num, one_num)))
    I0 = np.matrix(np.identity(zero_num))
    I1 = np.matrix(np.identity(one_num))
    C0 = I0 - J0 / zero_num
    C1 = I1 - J1 / one_num
    S0 = X0 * C0 * X0.T / zero_num
    S1 = X1 * C1 * X1.T / one_num
    lam0 = zero_num * np.log2(np.linalg.det(S0))
    lam1 = one_num * np.log2(np.linalg.det(S1))

    MI = (lam - lam0 - lam1) / n / 2
    return MI


def mutual_information_multinormal_gradient(F, Y, W, b):
    zero_num = np.sum(Y == 0)
    one_num = np.sum(Y == 1)
    n = F.shape[0]
    assert n == zero_num + one_num
    X = np.matrix(F).T
    J = np.matrix(np.ones((n, n)))
    I = np.matrix(np.identity(n))
    C = I - J / n
    S = X * C * X.T / n
    lam = n * np.log2(np.linalg.det(S))

    X0 = X * I[:, : zero_num]
    X1 = X * I[:, zero_num :]
    J0 = np.matrix(np.ones((zero_num, zero_num)))
    J1 = np.matrix(np.ones((one_num, one_num)))
    I0 = np.matrix(np.identity(zero_num))
    I1 = np.matrix(np.identity(one_num))
    C0 = I0 - J0 / zero_num
    C1 = I1 - J1 / one_num
    S0 = X0 * C0 * X0.T / zero_num
    S1 = X1 * C1 * X1.T / one_num
    lam0 = zero_num * np.log2(np.linalg.det(S0))
    lam1 = one_num * np.log2(np.linalg.det(S1))

    A = 2 / n * (S.I * X * C - S0.I * X * I[:, : zero_num] * C0 * I[:, : zero_num].T - S1.I * X * I[:, zero_num :] * C1 * I[:, zero_num :].T)
    grad = 0
    for i, Wi in enumerate(W):
        grad += np.matrix(Wi).T * A * I[:, i]
    
    return np.squeeze(np.asarray(grad))
