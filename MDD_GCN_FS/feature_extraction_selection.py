import os
import glob
import argparse
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse.linalg import eigsh


def spectral_filter_matrices(sample_path, filter_kernel):
    sub_dirs = [x[0] for x in os.walk(sample_path)]
    sub_dirs.pop(0)
    filters = scipy.sparse.csr_matrix((90, 90))

    for sub_dir in sub_dirs:
        file_list = []
        dir_name = os.path.basename(sub_dir)
        file_glob = os.path.join(sample_path, dir_name, '*')
        file_list.extend(glob.glob(file_glob))

        for f in file_list:
            dataframe = pd.read_csv(f, sep="\s+", header=None)
            w = dataframe.to_numpy()
            spec_f = spectral_filter(w, filter_kernel)
            if filters.size == 0:
                filters = spec_f
            else:
                filters = scipy.sparse.vstack(filters, spec_f)

    return filters      # (90*sample_num, 90)


def spectral_filter(w, ks):
    M, M = w.shape
    w = scipy.sparse.csr_matrix(w)
    L = laplacian(w, normalized=True)
    L = rescaled_L(L)
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

    return L_list[-1]


def laplacian(W, normalized=True):
    """Return graph Laplacian"""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L


def rescaled_L(laplacian):
    M, M = laplacian.shape
    I = scipy.sparse.identity(M, dtype=laplacian.dtype, format='csr')
    laplacian /= Lambda_max(laplacian) / 2
    laplacian -= I
    return laplacian


def Lambda_max(laplacian):
    return eigsh(laplacian, k=1, which='LM', return_eigenvectors=False)[0]


def stimulus_construction(w, type):
    if type == "constant":
        return np.ones(w.shape, dtype=w.dtype) 
    elif type == "linear":
        pass


def main():
    parser = argparse.ArgumentParser(description="feature extraction and selection")
    parser.add_argument('-DP', '--datapath', default='D:/code/DTI_data/network_FN/')
    parser.add_argument('-K', '--kernel', type=int, default=1, metavar='K')
    args = parser.parse_args()

    stimulus = np.random.uniform(0, 1, 90)
    stimulus = stimulus / np.linalg.norm(stimulus)
    stimulus = scipy.sparse.csr_matrix(np.expand_dims(stimulus, axis=1))

    filters = spectral_filter_matrices(args.datapath, filter_kernel=args.kernel)
    response = filters * stimulus
    response = response.reshape(-1, 90)     # every row represents a sample
    response.toarray()
    df = pd.DataFrame(response)
    df.to_csv("D:/code/DTI_data/output/", header=False, index=False)


if __name__ == "__main__":
    main()
