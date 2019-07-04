import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


OUT_PATH = "D:/code/out/"


df = pd.read_csv(OUT_PATH + "dist_A.csv", index_col=0)
label = pd.read_csv(OUT_PATH + "AAL_LabelID_116.txt", sep="\s+", index_col=0, header=None)

df.set_axis(label[1], axis=0)
df.set_axis(label[1], axis=1)

f, ax = plt.subplots(figsize=(33, 27))

ax = sns.heatmap(df, xticklabels=True, yticklabels=True, cbar_kws={'label': 'Spacial Connection Intensity'})

ax.set_xlabel("")
ax.set_ylabel("")
ax.figure.axes[-1].yaxis.label.set_size(50)
ax.figure.axes[-1].yaxis.labelpad = 40
ax.figure.axes[-1].set_yticklabels(ax.figure.axes[-1].get_ymajorticklabels(), fontsize=40)
plt.savefig(OUT_PATH + "dist_A.png")


df = pd.read_csv(OUT_PATH + "A_0.csv", index_col=0)
f, ax = plt.subplots(figsize=(22, 18))
ax = sns.heatmap(df)
plt.savefig(OUT_PATH + "A_0.png")

df = pd.read_csv(OUT_PATH + "A_1.csv", index_col=0)
f, ax = plt.subplots(figsize=(22, 18))
ax = sns.heatmap(df)
plt.savefig(OUT_PATH + "A_1.png")

df = pd.read_csv(OUT_PATH + "A_2.csv", index_col=0)
f, ax = plt.subplots(figsize=(22, 18))
ax = sns.heatmap(df)
plt.savefig(OUT_PATH + "A_2.png")

df = pd.read_csv(OUT_PATH + "A_3.csv", index_col=0)
f, ax = plt.subplots(figsize=(22, 18))
ax = sns.heatmap(df)
plt.savefig(OUT_PATH + "A_3.png")

df = pd.read_csv(OUT_PATH + "A_4.csv", index_col=0)
f, ax = plt.subplots(figsize=(22, 18))
ax = sns.heatmap(df)
plt.savefig(OUT_PATH + "A_4.png")


# output files for BrainNet

# node-116
df = pd.read_csv(OUT_PATH + "node_perm.csv", index_col=0)
perm = pd.read_csv(OUT_PATH + "perm.csv", index_col=0)
perm_list = np.squeeze(perm.values).tolist()
label_list = label.values.tolist()

label_perm_list = []
for idx in perm_list:
    if idx < len(label_list):
        label_perm_list.append(label_list[idx])

label_perm = pd.DataFrame(label_perm_list)

zeros = pd.DataFrame(np.zeros(len(label_list), dtype=int))
color = np.array(range(len(label_list)), dtype=float)
color[::2] = color[::2] + 0.4
color[1::2] = color[1::2] - 0.4
color = pd.DataFrame(color)
zeros.columns = ["z"]
color.columns = ["color"]
label_perm.columns = ["label"]

df = df.join(color)
df = df.join(zeros)
df = df.join(label_perm)
df.to_csv(OUT_PATH + "AAL_116_perm.node", sep="\t", header=False, index=False)



# weight-116
weight = pd.read_csv(OUT_PATH + "A_0.csv", index_col=0)
weight = weight.to_numpy()
weight = weight[:,~np.all(weight == 0, axis=0)]
weight = weight[~np.all(weight == 0, axis=1)]
df = pd.DataFrame(weight)
df.to_csv(OUT_PATH + "weight_116_perm.edge", sep="\t", header=False, index=False)

# weight-58
weight = pd.read_csv(OUT_PATH + "A_0.csv", index_col=0)
weight = weight.to_numpy()

node = pd.read_csv(OUT_PATH + "AAL_116_perm.node", sep="\t", header=None)
node = node.iloc[:, 0:3].to_numpy()

node_coar = np.empty([64, 3])
header = 0
for i in range(weight.shape[0] // 2):
    if np.all(weight[2*i] == 0, axis=0) and np.all(weight[2*i+1] == 0, axis=0):
        node_coar[i] = np.zeros_like(node[0])
    elif not (np.all(weight[2*i] == 0, axis=0) or np.all(weight[2*i+1] == 0, axis=0)):
        node_coar[i] = (node[header] + node[header+1]) / 2
        header += 2
    else:
        node_coar[i] = node[header]
        header += 1
node_coar = node_coar[~np.all(node_coar == 0 ,axis=1)]
node_coar = np.around(node_coar, decimals=2)
df = pd.DataFrame(node_coar)

color = np.array(range(node_coar.shape[0]), dtype=float)
color[::2] = color[::2] + 0.4
color[1::2] = color[1::2] - 0.4
color = pd.DataFrame(color)
color.columns = ["color"]
zeros = pd.DataFrame(np.zeros(node_coar.shape[0], dtype=int), columns=["z"])
label = pd.DataFrame(np.zeros(node_coar.shape[0], dtype=int), columns=["label"])
df = df.join(color)
df = df.join(zeros)
df = df.join(label)

df.to_csv(OUT_PATH + "AAL_" + str(node_coar.shape[0]) + "_perm.node", sep="\t", header=False, index=False)

weight = pd.read_csv(OUT_PATH + "A_1.csv", index_col=0)
weight = weight.to_numpy()
weight = weight[:,~np.all(weight == 0, axis=0)]
weight = weight[~np.all(weight == 0, axis=1)]
df = pd.DataFrame(weight)
df.to_csv(OUT_PATH + "weight_" + str(weight.shape[0]) + "_perm.edge", sep="\t", header=False, index=False)


# weight-29
weight = pd.read_csv(OUT_PATH + "A_1.csv", index_col=0)
weight = weight.to_numpy()

node = pd.read_csv(OUT_PATH + "AAL_58_perm.node", sep="\t", header=None)
node = node.iloc[:, 0:3].to_numpy()

node_coar = np.empty([32, 3])
header = 0
for i in range(weight.shape[0] // 2):
    if np.all(weight[2*i] == 0, axis=0) and np.all(weight[2*i+1] == 0, axis=0):
        node_coar[i] = np.zeros_like(node[0])
    elif not (np.all(weight[2*i] == 0, axis=0) or np.all(weight[2*i+1] == 0, axis=0)):
        node_coar[i] = (node[header] + node[header+1]) / 2
        header += 2
    else:
        node_coar[i] = node[header]
        header += 1
node_coar = node_coar[~np.all(node_coar == 0 ,axis=1)]
node_coar = np.around(node_coar, decimals=2)
df = pd.DataFrame(node_coar)

color = np.array(range(node_coar.shape[0]), dtype=float)
color[::2] = color[::2] + 0.4
color[1::2] = color[1::2] - 0.4
color = pd.DataFrame(color)
color.columns = ["color"]
zeros = pd.DataFrame(np.zeros(node_coar.shape[0], dtype=int), columns=["z"])
label = pd.DataFrame(np.zeros(node_coar.shape[0], dtype=int), columns=["label"])
df = df.join(color)
df = df.join(zeros)
df = df.join(label)

df.to_csv(OUT_PATH + "AAL_" + str(node_coar.shape[0]) + "_perm.node", sep="\t", header=False, index=False)

weight = pd.read_csv(OUT_PATH + "A_2.csv", index_col=0)
weight = weight.to_numpy()
weight = weight[:,~np.all(weight == 0, axis=0)]
weight = weight[~np.all(weight == 0, axis=1)]
df = pd.DataFrame(weight)
df.to_csv(OUT_PATH + "weight_" + str(weight.shape[0]) + "_perm.edge", sep="\t", header=False, index=False)


# weight-15
weight = pd.read_csv(OUT_PATH + "A_2.csv", index_col=0)
weight = weight.to_numpy()

node = pd.read_csv(OUT_PATH + "AAL_29_perm.node", sep="\t", header=None)
node = node.iloc[:, 0:3].to_numpy()

node_coar = np.empty([16, 3])
header = 0
for i in range(weight.shape[0] // 2):
    if np.all(weight[2*i] == 0, axis=0) and np.all(weight[2*i+1] == 0, axis=0):
        node_coar[i] = np.zeros_like(node[0])
    elif not (np.all(weight[2*i] == 0, axis=0) or np.all(weight[2*i+1] == 0, axis=0)):
        node_coar[i] = (node[header] + node[header+1]) / 2
        header += 2
    else:
        node_coar[i] = node[header]
        header += 1
node_coar = node_coar[~np.all(node_coar == 0 ,axis=1)]
node_coar = np.around(node_coar, decimals=2)
df = pd.DataFrame(node_coar)

color = np.array(range(node_coar.shape[0]), dtype=float)
color[::2] = color[::2] + 0.4
color[1::2] = color[1::2] - 0.4
color = pd.DataFrame(color)
color.columns = ["color"]
zeros = pd.DataFrame(np.zeros(node_coar.shape[0], dtype=int), columns=["z"])
label = pd.DataFrame(np.zeros(node_coar.shape[0], dtype=int), columns=["label"])
df = df.join(color)
df = df.join(zeros)
df = df.join(label)

df.to_csv(OUT_PATH + "AAL_" + str(node_coar.shape[0]) + "_perm.node", sep="\t", header=False, index=False)

weight = pd.read_csv(OUT_PATH + "A_3.csv", index_col=0)
weight = weight.to_numpy()
weight = weight[:,~np.all(weight == 0, axis=0)]
weight = weight[~np.all(weight == 0, axis=1)]
df = pd.DataFrame(weight)
df.to_csv(OUT_PATH + "weight_" + str(weight.shape[0]) + "_perm.edge", sep="\t", header=False, index=False)


# weight-8
weight = pd.read_csv(OUT_PATH + "A_3.csv", index_col=0)
weight = weight.to_numpy()

node = pd.read_csv(OUT_PATH + "AAL_15_perm.node", sep="\t", header=None)
node = node.iloc[:, 0:3].to_numpy()

node_coar = np.empty([8, 3])
header = 0
for i in range(weight.shape[0] // 2):
    if np.all(weight[2*i] == 0, axis=0) and np.all(weight[2*i+1] == 0, axis=0):
        node_coar[i] = np.zeros_like(node[0])
    elif not (np.all(weight[2*i] == 0, axis=0) or np.all(weight[2*i+1] == 0, axis=0)):
        node_coar[i] = (node[header] + node[header+1]) / 2
        header += 2
    else:
        node_coar[i] = node[header]
        header += 1
node_coar = node_coar[~np.all(node_coar == 0 ,axis=1)]
node_coar = np.around(node_coar, decimals=2)
df = pd.DataFrame(node_coar)

color = np.array(range(node_coar.shape[0]), dtype=float)
color[::2] = color[::2] + 0.4
color[1::2] = color[1::2] - 0.4
color = pd.DataFrame(color)
color.columns = ["color"]
zeros = pd.DataFrame(np.zeros(node_coar.shape[0], dtype=int), columns=["z"])
label = pd.DataFrame(np.zeros(node_coar.shape[0], dtype=int), columns=["label"])
df = df.join(color)
df = df.join(zeros)
df = df.join(label)

df.to_csv(OUT_PATH + "AAL_" + str(node_coar.shape[0]) + "_perm.node", sep="\t", header=False, index=False)

weight = pd.read_csv(OUT_PATH + "A_4.csv", index_col=0)
weight = weight.to_numpy()
weight = weight[:,~np.all(weight == 0, axis=0)]
weight = weight[~np.all(weight == 0, axis=1)]
df = pd.DataFrame(weight)
df.to_csv(OUT_PATH + "weight_" + str(weight.shape[0]) + "_perm.edge", sep="\t", header=False, index=False)

