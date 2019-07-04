import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

# Generate a large random dataset
OUT_PATH = "D:/code/out/"

df = pd.read_csv(OUT_PATH + "dist_A.csv", index_col=0)
label = pd.read_csv(OUT_PATH + "AAL_LabelID_116.txt", sep="\s+", index_col=0, header=None)

df.set_axis(label[1], axis=0)
df.set_axis(label[1], axis=1)
"""
grid_spec = {"width_ratios": (.9, .05)}
f, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_spec) 
sns.heatmap(df, ax=ax, cbar_ax=cbar_ax, cbar_kws={'label': 'spacial connection intensity'})
cbar_ax.set_yticklabels(cbar_ax.get_ymajorticklabels(), fontsize=100)
plt.show()"""
# Set up the matplotlib figure

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
df = pd.read_csv(OUT_PATH + "node_perm.csv", index_col=0)
df = 