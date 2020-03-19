import numpy as np

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
    

    # x0_mu = np.array([15, 7, 12, 8])
    # x0_cov = np.array([[3, 0.2, 0.5, 0.3], [0.2, 1, 0, 0.1], [0.5, 0, 2, 0.5], [0.3, 0.1, 0.5, 2]])
    # # x0_cov = np.array([[1, 0.2, 0.5], [0.2, 1, 0], [0.5, 0, 1]])
    # x1_mu = np.array([13, 8, 11, 10])
    # x1_cov = np.array([[1, 0.6, 1, 0.2], [0.6, 0.6, 0, 0.1], [1, 0, 3, 0.3], [0.2, 0.1, 0.3, 3]])
    # x1_cov = np.array([[1, 0.6, 0.9], [0.6, 0.6, 0], [0.9, 0, 3]])

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


if __name__ == "__main__":
    w, y, _ = graph_generate(4, 100, random_seed=100)
    print()