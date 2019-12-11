import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def node_select(PATH):
    df = pd.read_csv(PATH, header=None)
    activate = df.to_numpy()
    ii = np.unravel_index(np.argsort(activate.ravel())[-90:], activate.shape)    # get the indices of top 90
    iix = np.expand_dims(ii[0], axis=1)
    iiy = np.expand_dims(ii[1], axis=1)
    iixy = np.concatenate((iix, iiy), axis=1)

    return ii


PicRoot = "D:/code/trans_pic/"

# ---------------- figure 2 ---------------- #
# PATH1 = "D:/code/DTI_data/result/pre-train_cnn_acc.csv"
# PATH2 = "D:/code/DTI_data/result/pre-train_mix_acc.csv"
# df1 = pd.read_csv(PATH1, header=None, index_col=0)
# df2 = pd.read_csv(PATH2, header=None, index_col=0)
# acc1 = df1[1].tolist()
# acc2 = df2[1].tolist()
# acc1.insert(0,0.5)
# acc2.insert(0,0.5)
# plt.plot(list(range(df1.shape[0]+1)), acc1, 'k-', label='conventional CNN')
# plt.plot(list(range(df2.shape[0]+1)), acc2, 'k--', label='mixed CNN')
# plt.axis([0, df1.shape[0], 0.5, 1])
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.legend(loc=4)
# plt.savefig(PicRoot + 'fig2.eps')
# plt.show()

# ---------------- figure 3 ---------------- #

# PATH = "D:/code/DTI_data/pretrain/191129_sr_0.001_var/epoch_30.csv"
# df = pd.read_csv(PATH, header=None)

# f, ax = plt.subplots(figsize=(6,6))

# ax = sns.heatmap(df, annot=True, fmt='.1f', vmin=0, vmax=1, annot_kws={"fontsize":8}, cbar=False)
# ax.tick_params(left=False, bottom=False)
# plt.savefig(PicRoot + 'fig3.eps')
# plt.show()

# ---------------- figure 4 ---------------- #

df = pd.DataFrame(
    [
        [0.580, 0.485, 0.615],
        [0.538, 0.455, 0.570],
        [0.545, 0.523, 0.600],
        [0.552, 0.558, 0.560],
        [0.570, 0.468, 0.583]
    ],  columns=['SVM', 'FTGCNN', 'PTGCNN']
)
# f, ax = plt.subplots(figsize=(6,6))
ax = sns.boxplot(data=df, showmeans=True, width=0.5, color="skyblue",
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"})
plt.ylabel('accuracy')
plt.savefig(PicRoot + 'fig4.jpg')
plt.show()

# ---------------- figure other ---------------- #

# df = pd.DataFrame(
#     [
#         [0.619, 0.485, 0.529],
#         [0.616, 0.455, 0.570],
#         [0.633, 0.523, 0.600],
#         [0.618, 0.558, 0.560],
#         [0.629, 0.468, 0.583]
#     ],  columns=['SVM', 'FTGCNN', 'PTGCNN']
# )
# # f, ax = plt.subplots(figsize=(6,6))
# ax = sns.boxplot(data=df)
# plt.ylabel('accuracy')
# plt.savefig(PicRoot + 'fig4.eps')
# plt.show()