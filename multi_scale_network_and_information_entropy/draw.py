import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ggmfit import *
from mutual_information import *
from data_loader import *
from PSO import *
from MI_learning import *


# # -------------------MI convergence------------------- #
# path = "D:/code/mutual_information_toy_output/different_num_particle_MIcurve/"
# fitlist = pd.read_csv(path + 'w1_dim_6_samplenum_2000_partnum_20_fitlist.csv', header=None, index_col=None)
# MIlist = pd.read_csv(path + 'w1_dim_6_samplenum_2000_partnum_20_MIlist.csv', index_col=None)

# fitlist = fitlist.to_numpy()
# fitlist = np.delete(fitlist, 0, axis=1) 
# MI = MIlist.iloc[0,3].replace('\n', '')
# MI = MI.replace('  ', ' ')
# MI = MI.replace(' ', ', ')
# MI = np.array(eval(MI))

# for fit in fitlist:
#     plt.plot(fit)
# MI_p = plt.plot(MI, "--", color="red", linewidth=4.0, label="Global best mutual information")
# # plt.legend(handles=[MI_p],labels=["Global best mutual information"],loc='lower right')
# plt.show()
# print()


# # -------------------different number of particles convergence------------------- #
# path = "D:/code/mutual_information_toy_output/different_num_particle_MIcurve/"
# MIlist_5 = pd.read_csv(path + 'w1_dim_6_samplenum_2000_partnum_5_MIlist.csv', index_col=None)
# MIlist_10 = pd.read_csv(path + 'w1_dim_6_samplenum_2000_partnum_10_MIlist.csv', index_col=None)
# MIlist_20 = pd.read_csv(path + 'w1_dim_6_samplenum_2000_partnum_20_MIlist.csv', index_col=None)
# MIlist_30 = pd.read_csv(path + 'w1_dim_6_samplenum_2000_partnum_30_MIlist.csv', index_col=None)

# MI_5 = MIlist_5.iloc[0,3].replace('\n', '')
# MI_5 = MI_5.replace('  ', ' ')
# MI_5 = MI_5.replace(' ', ', ')
# MI_5 = np.array(eval(MI_5))

# MI_10 = MIlist_10.iloc[0,3].replace('\n', '')
# MI_10 = MI_10.replace('  ', ' ')
# MI_10 = MI_10.replace(' ', ', ')
# MI_10 = np.array(eval(MI_10))

# MI_20 = MIlist_20.iloc[0,3].replace('\n', '')
# MI_20 = MI_20.replace('  ', ' ')
# MI_20 = MI_20.replace(' ', ', ')
# MI_20 = np.array(eval(MI_20))

# MI_30 = MIlist_30.iloc[0,3].replace('\n', '')
# MI_30 = MI_30.replace('    ', ' ')
# MI_30 = MI_30.replace('   ', ' ')
# MI_30 = MI_30.replace('  ', ' ')
# MI_30 = MI_30.replace(' ', ', ')
# MI_30 = np.array(eval(MI_30))

# line_5, = plt.plot(MI_5, label='5 particles')
# line_10, = plt.plot(MI_10, label='10 particles')
# line_20, = plt.plot(MI_20, label='20 particles')
# line_30, = plt.plot(MI_30, label='30 particles')
# plt.legend(loc='lower right')
# plt.axis([0, 200, 0.0, 1.5])
# plt.savefig(path + 'different_num_particles_convergence.png')
# plt.show()

# print()


# # -------------------different number of particles convergence------------------- #
# path = "D:/code/mutual_information_toy_output/"
# MIlist_20 = pd.read_csv(path + 'shape_6_k_6_sparserate_0.2_MI_list_cv_0.csv', header=None, index_col=None)
# MIlist_20 = MIlist_20.to_numpy()


# line_1, = plt.plot(MIlist_20[0], label='radius: 1')
# line_2, = plt.plot(MIlist_20[1], label='radius: 2')
# line_3, = plt.plot(MIlist_20[2], label='radius: 3')
# line_4, = plt.plot(MIlist_20[3], label='radius: 4')
# line_5, = plt.plot(MIlist_20[4], label='radius: 5')
# line_6, = plt.plot(MIlist_20[5], label='radius: 6')

# plt.legend(loc='lower right')
# plt.axis([0, 200, 0.0, 1.8])
# plt.savefig(path + 'different_radius_convergence.png')
# plt.show()

# print()

# -------------------MIfeaure selection------------------- #
path = "D:/code/mutual_information_toy_output/"
b_list = pd.read_csv(path + 'shape_6_k_6_sparserate_0.2_b_list_cv_0.csv', header=None, index_col=None)
b_list = b_list.to_numpy()

shape = 6
sample_num = 2000
np.random.seed(123)
random_seed = 123456

# MI learning parameter
k = 6
sparse_rate = 0.2

x, y, _ = nxnetwork_generate(shape, sample_num, random_seed)
g = graph_mine(x, y, sparse_rate)

# 10-fold validation
cv = 10
kf = KFold(n_splits=cv, shuffle=True, random_state=random_seed)
acc_sum = 0
for idx, (train_idx, test_idx) in enumerate(kf.split(x)):
    if idx != 0:
        continue
    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    fs_num = 10
    pca_num = 10

    f1_train = []
    f1_test = []
    for b in b_list:
        f_train = np.matmul(x_train, b)
        f1_train.append(f_train)
        f_test = np.matmul(x_test, b)
        f1_test.append(f_test)
    f1_train = np.array(f1_train)
    f1_train = np.transpose(f1_train, (1, 0, 2))
    f1_train = np.reshape(f1_train, (f1_train.shape[0], -1))
    f1_test = np.array(f1_test)
    f1_test = np.transpose(f1_test, (1, 0, 2))
    f1_test = np.reshape(f1_test, (f1_test.shape[0], -1))

    # fisher scoring
    class0 = f1_train[y_train == 0]
    class1 = f1_train[y_train == 1]
    Mu0 = np.mean(class0, axis=0)
    Mu1 = np.mean(class1, axis=0)
    Mu = np.mean(f1_train, axis=0)
    Sigma0 = np.var(class0, axis=0)
    Sigma1 = np.var(class1, axis=0)
    n0 = class0.shape[0]
    n1 = class1.shape[0]
    fisher_score = (n0 * (Mu0 - Mu)**2 + n1 * (Mu1 - Mu)**2) / (n0 * Sigma0 + n1 * Sigma1)
    sort_idx = np.argsort(fisher_score)[::-1]
    value = np.arange(0, 36)
    value = value + 1
    sort_value = np.zeros(36)
    sort_value[sort_idx] = value
    sort_value = np.reshape(sort_value, (6, 6)).T

    f, ax = plt.subplots(figsize=(6,6))

    ax = sns.heatmap(sort_value, annot=True, vmin=1, vmax=36, annot_kws={"fontsize":14}, cbar=False)
    ax.tick_params(left=False, bottom=False)
    plt.savefig(path + 'feature_selection.eps')
    plt.show()



# # -------------------MI rate validation------------------- #
# path = "D:/code/mutual_information_toy_output/"
# b_list = pd.read_csv(path + 'shape_6_k_6_sparserate_0.2_b_list_cv_0.csv', header=None, index_col=None)



# shape = 6
# sample_num = 2000
# np.random.seed(123)
# random_seed = 123456

# # MI learning parameter
# k = 6
# sparse_rate = 0.2

# x, y, _ = nxnetwork_generate(shape, sample_num, random_seed)
# g = graph_mine(x, y, sparse_rate)


# b1 = [0.57663745, -0.63329087, 0.0947998, -0.47462473, 0.0842422, 0.15836521]
# b2 = [0.365277552, 0.565101509, 0.643381858, -0.169381208, 0.323166489, 0.012876536]
# b_best = b_list.to_numpy()[0]

# # 10-fold validation
# cv = 10
# kf = KFold(n_splits=cv, shuffle=True, random_state=random_seed)
# acc_sum = 0
# for idx, (train_idx, test_idx) in enumerate(kf.split(x)):
#     if idx != 0:
#         continue
#     x_train = x[train_idx]
#     x_test = x[test_idx]
#     y_train = y[train_idx]
#     y_test = y[test_idx]

#     fs_num = 6
#     pca_num = 6

#     f1_train = []
#     f1_test = []
#     b = b_best
#     f1_train = np.matmul(x_train, b)
#     f1_test = np.matmul(x_test, b)

#     # Norm
#     scaler = StandardScaler()
#     scaler.fit(f1_train)
#     f1scale_train = scaler.transform(f1_train)
#     f1scale_test = scaler.transform(f1_test)

#     # SVC
#     svc = SVC(kernel='rbf', random_state=1, gamma=0.001, C=1000)
#     model = svc.fit(f1scale_train, y_train)

#     predict_train = model.predict(f1scale_train)
#     correct_train = np.sum(predict_train == y_train)
#     accuracy_train = correct_train / train_idx.size

#     predict = model.predict(f1scale_test)
#     correct = np.sum(predict == y_test)
#     print()
#     print(classification_report(y_test, predict))
#     accuracy = correct / test_idx.size
#     print("cv: {}/{}, acc.: {:.1f}/{:.1f}\n".format(idx, cv, accuracy_train*100, accuracy*100))
#     acc_sum += accuracy
#     print("total acc.: {:.1f}\n".format(acc_sum / cv * 100))
#     print()


# ----dif rate---#
index=np.array([1,2,3,4,5,6])
a = np.array([0.95, 0.945, 0.932, 0.928, 0.92, 0.928])
# width = 0.45
plt.plot(index, a)
plt.axis([0, 7, 0.0, 1.0])
# plt.bar(index, a, width)
plt.show()