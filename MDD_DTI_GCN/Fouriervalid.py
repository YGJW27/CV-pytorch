import pandas as pd
import numpy as np
import scipy.sparse
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

out_path = "D:/code/out/"

weight = pd.read_csv(out_path + "A_0.csv", index_col=0)
weight_n = weight.to_numpy()
weight_s = scipy.sparse.coo_matrix(weight_n)
node_i = pd.DataFrame(np.expand_dims(weight_s.row, axis=1), columns=["i"])
node_j = pd.DataFrame(np.expand_dims(weight_s.col, axis=1), columns=["j"])
weight_ij = pd.DataFrame(np.expand_dims(weight_s.data, axis=1), columns=["weight"])
df = node_i.join(node_j)
df = df.join(weight_ij)
G = nx.from_pandas_edgelist(df, source='i', target='j', edge_attr=True)

length = dict(nx.shortest_path_length(G))
neighbor = np.zeros(weight.shape, dtype=int)
for i in length.keys():
    for j in length[i].keys():
        neighbor[i][j] = length[i][j]           # path length from row i to column j

L = pd.read_csv(out_path + "rescaled_L0.csv", index_col=0)
L_1 = L.to_numpy()
L_2 = 2 * np.dot(L_1, L_1) - np.diag(np.ones(L_1.shape[0]))
L_3 = 2 * np.dot(L_1, L_2) - L_1
L_4 = 2 * np.dot(L_1, L_3) - L_2

L_1[L_1 != 0] = 1
L_2[L_2 != 0] = 2
L_3[L_3 != 0] = 3
L_4[L_4 != 0] = 4
np.fill_diagonal(L_1, 0)
np.fill_diagonal(L_2, 0)
np.fill_diagonal(L_3, 0)
np.fill_diagonal(L_4, 0)

L_itg = L_4
L_itg[L_3 != 0] = 3
L_itg[L_2 != 0] = 2
L_itg[L_1 != 0] = 1

assert (L_itg == neighbor).all()               # True!

assert (L_2 >= L_1).all()
assert (L_3 >= L_2).all()
assert (L_4 >= L_3).all()

df = pd.DataFrame(neighbor)
f, ax = plt.subplots(figsize=(22, 18))
ax = sns.heatmap(df)
plt.savefig(out_path + "A_0_pathlen.png")

df = pd.DataFrame(L_1)
f, ax = plt.subplots(figsize=(22, 18))
ax = sns.heatmap(df)
plt.savefig(out_path + "L_1.png")

df = pd.DataFrame(L_2)
f, ax = plt.subplots(figsize=(22, 18))
ax = sns.heatmap(df)
plt.savefig(out_path + "L_2.png")

df = pd.DataFrame(L_3)
f, ax = plt.subplots(figsize=(22, 18))
ax = sns.heatmap(df)
plt.savefig(out_path + "L_3.png")

df = pd.DataFrame(L_4)
f, ax = plt.subplots(figsize=(22, 18))
ax = sns.heatmap(df)
plt.savefig(out_path + "L_4.png")
