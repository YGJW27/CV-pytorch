import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def main_1():
    output_path = 'D:/code/mutual_information_toy_output/'
    shape = 6
    sample_num = 2000
    random_seed = 123456
    df = pd.DataFrame(columns=['part_num', 'iter_num', 'b', 'MI'])
    for randi in range(10):
        w, y, _ = nxnetwork_generate(shape, sample_num, random_seed+randi)

        G = graph_mine(w, y, 0.3)

        starttime = time.time()
        # part_num_array = [5, 10, 20, 30]
        part_num_array = [20]
        for part_num in part_num_array:
            iter_num = 2000
            omega_max = 0.9
            omega_min = 0.4
            c1 = 2
            c2 = 2

            fitness_func = fitness(w, y, G)
            b, MI_array, fit_list = PSO(fitness_func, part_num, iter_num, omega_max, omega_min, c1, c2)
            endtime = time.time()
            runtime = endtime - starttime
            print('time: {:.2f}'.format(runtime))
            print(b, MI_array[-1])

            # fit_df = pd.DataFrame(fit_list)
            # fit_df.to_csv(output_path + \
            #     'w1_dim_{:d}_samplenum_{:d}_partnum_30_fitlist.csv'.format(shape, sample_num),
            #     header=False,
            #     index=False
            #     )

            df = df.append(pd.DataFrame(np.array([part_num, iter_num, b, MI_array]).reshape(1, -1), columns=['part_num', 'iter_num', 'b', 'MI']))
            plt.plot(MI_array)

    df.to_csv(output_path + \
        'w1_dim_{:d}_samplenum_{:d}_partnum_20_MIlist_10times.csv'.format(shape, sample_num),
        index=False
        )
    plt.savefig(output_path + \
        'w1_dim_{:d}_samplenum_{:d}_partnum_20_MIlist_10times.png'.format(shape, sample_num),
        )
    plt.show()


def main_2():
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

        fs_num = 10
        pca_num = 10

        # model = MI_learning(x_train, y_train, g, k)
        # b_list, MI_list, b_track_list = model.learning(part_num, iter_num, omega_max, omega_min, c1, c2)

        # b_track_df = pd.DataFrame(b_track_list[0])
        # b_track_df.to_csv(output_path + 'shape_{:d}_k_1_sparserate_{:.1f}_b_track_cv_{:d}.csv'.format(shape, sparse_rate, idx), header=False,
        #     index=False
        #     )

        # b_df = pd.DataFrame(b_list)
        # MI_df = pd.DataFrame(MI_list)
        # b_df.to_csv(output_path + 'shape_{:d}_k_{:d}_sparserate_{:.1f}_b_list_cv_{:d}.csv'.format(shape, k, sparse_rate, idx), header=False,
        #     index=False
        #     )
        # MI_df.to_csv(output_path + 'shape_{:d}_k_{:d}_sparserate_{:.1f}_MI_list_cv_{:d}.csv'.format(shape, k, sparse_rate, idx), header=False,
        #     index=False
        #     )


        # f1_train = []
        # f1_test = []
        # for b in b_list:
        #     f_train = np.matmul(x_train, b)
        #     f1_train.append(f_train)
        #     f_test = np.matmul(x_test, b)
        #     f1_test.append(f_test)
        # f1_train = np.array(f1_train)
        # f1_train = np.transpose(f1_train, (1, 0, 2))
        # f1_train = np.reshape(f1_train, (f1_train.shape[0], -1))
        # f1_test = np.array(f1_test)
        # f1_test = np.transpose(f1_test, (1, 0, 2))
        # f1_test = np.reshape(f1_test, (f1_test.shape[0], -1))

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

        # # Set the parameters by cross-validation
        # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],
        #                     'C': [1, 10, 100, 1000]},
        #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        # scores = ['precision', 'recall']

        # for score in scores:
        #     print("# Tuning hyper-parameters for %s" % score)
        #     print()

        #     clf = GridSearchCV(
        #         SVC(), tuned_parameters, scoring='%s_macro' % score
        #     )
        #     clf.fit(f1scale_train, y_train)

        #     print("Best parameters set found on development set:")
        #     print()
        #     print(clf.best_params_)
        #     print()
        #     print("Grid scores on development set:")
        #     print()
        #     means = clf.cv_results_['mean_test_score']
        #     stds = clf.cv_results_['std_test_score']
        #     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #         print("%0.3f (+/-%0.03f) for %r"
        #             % (mean, std * 2, params))
        #     print()

        #     print("Detailed classification report:")
        #     print()
        #     print("The model is trained on the full development set.")
        #     print("The scores are computed on the full evaluation set.")
        #     print()
        #     y_true, y_pred = y_test, clf.predict(f1scale_test)
        #     print(classification_report(y_true, y_pred))
        #     print()

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
    main_2()