import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from ggmfit import *
from mutual_information import *
from data_loader import *
from PSO import *
from MI_learning import *


def plot_embedding(X, Y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1], marker=('^' if Y[i] else 'o'),
        edgecolors=plt.cm.Set1(Y[i]),
        color='')

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     shown_images = np.array([[1., 1.]])
    #     for i in range(X.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
    #             X[i])
    #         ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    if title is not None:
        plt.title(title)


def main():
    output_path = "D:/code/mutual_information_toy_output/"
    shape = 6
    sample_num = 2000
    np.random.seed(123)
    random_seed = 123456
    
    # MI learning parameter
    k = 6
    sparse_rate = 0.2

    x, y, _ = nxnetwork_generate(shape, sample_num, random_seed)
    g = graph_mine(x, y, sparse_rate)

    starttime = time.time()

    # PSO parameters
    part_num = 20
    iter_num = 2000
    omega_max = 0.9
    omega_min = 0.4
    c1 = 2
    c2 = 2


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

        fs_num = 36
        pca_num = 36


        b_df = pd.read_csv(output_path + 'shape_{:d}_k_{:d}_sparserate_{:.1f}_b_list_cv_{:d}.csv'.format(shape, k, sparse_rate, idx), header=None)
        b_list = b_df.to_numpy()

        f1_train = []
        f1_test = []
        for idx, b in enumerate(b_list):
            if idx != 5:
                continue
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


        # # read b_track_list
        # b_df = pd.read_csv(output_path + 'shape_{:d}_k_1_sparserate_{:.1f}_b_track_cv_{:d}.csv'.format(shape, sparse_rate, idx), header=None)
        # b_track = b_df.to_numpy()

        # for idx, b in enumerate(b_track):
        #     f1_train = np.matmul(x_train, b)
        #     f1_test = np.matmul(x_test, b)
        #     MI = mutual_information(f1_train, y_train, g)
        #     print(idx, MI)

        # np.random.seed(8)
        # b = np.random.uniform(size=6)
        # b = b/ np.linalg.norm(b)
        # b = b_track[2]

        # f1_train = np.matmul(x_train, b)
        # f1_test = np.matmul(x_test, b)
        # MI = mutual_information(f1_train, y_train, g)
        # print(idx, MI)



        # # fisher scoring
        # class0 = f1_train[y_train == 0]
        # class1 = f1_train[y_train == 1]
        # Mu0 = np.mean(class0, axis=0)
        # Mu1 = np.mean(class1, axis=0)
        # Mu = np.mean(f1_train, axis=0)
        # Sigma0 = np.var(class0, axis=0)
        # Sigma1 = np.var(class1, axis=0)
        # n0 = class0.shape[0]
        # n1 = class1.shape[0]
        # fisher_score = (n0 * (Mu0 - Mu)**2 + n1 * (Mu1 - Mu)**2) / (n0 * Sigma0 + n1 * Sigma1)
        # sort_idx = np.argsort(fisher_score)[::-1]
        # print("sort_idx: ", sort_idx)
        # print()
        # print("graph structure: ", g)
        # print()
        # sort_idx = sort_idx[:fs_num]
        
        # f1_train = f1_train[:, sort_idx]
        # f1_test = f1_test[:, sort_idx]


        # # matrix to array
        # f1_train = x_train.reshape(x_train.shape[0], -1)
        # f1_test = x_test.reshape(x_test.shape[0], -1)

        # Norm
        scaler = StandardScaler()
        scaler.fit(f1_train)
        f1scale_train = scaler.transform(f1_train)
        f1scale_test = scaler.transform(f1_test)

        # # PCA
        # pca = PCA(n_components=pca_num)
        # pca.fit(f1scale_train)
        # f1scale_train = pca.transform(f1scale_train)
        # f1scale_test = pca.transform(f1scale_test)

        # Draw
        X = f1scale_test.copy()
        X_draw = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X)
        plot_embedding(X_draw, y_test)
        plt.show()

        # # Draw
        # X = f1scale_train.copy()
        # X_draw = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X)
        # plot_embedding(X_draw, y_train)
        # plt.show()



        # # Ramdom Forest
        # rf = RandomForestClassifier(max_depth=5, random_state=0)
        # model = rf.fit(f1scale_train, y_train)

        # SVC
        svc = SVC(kernel='rbf', random_state=1, gamma=0.001, C=1000)
        model = svc.fit(f1scale_train, y_train)

        predict_train = model.predict(f1scale_train)
        correct_train = np.sum(predict_train == y_train)
        accuracy_train = correct_train / train_idx.size

        predict = model.predict(f1scale_test)
        correct = np.sum(predict == y_test)
        print()
        print(classification_report(y_test, predict))
        accuracy = correct / test_idx.size
        print("cv: {}/{}, acc.: {:.1f}/{:.1f}\n".format(idx, cv, accuracy_train*100, accuracy*100))
        acc_sum += accuracy
        print("total acc.: {:.1f}\n".format(acc_sum / cv * 100))
        print()

    endtime = time.time()
    runtime = endtime - starttime
    print('total time:', runtime, '\n')
    print()


if __name__ == "__main__":
    main()