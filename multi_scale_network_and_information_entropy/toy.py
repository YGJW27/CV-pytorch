import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ggmfit import *
from mutual_information import *
from data_loader import *
from PSO import *
from MI_learning import *


def main_1():
    output_path = 'D:/code/mutual_information_toy_output/'
    shape = 4
    sample_num = 2000
    random_seed = 123456
    df = pd.DataFrame(columns=['part_num', 'iter_num', 'b', 'MI'])
    for randi in range(1):
        w, y, _ = graph_generate(shape, sample_num, random_seed+randi)

        G = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        G = np.ones((shape, shape))

        starttime = time.time()
        part_num_array = [5, 10, 20, 30]
        for part_num in part_num_array:
            iter_num = 5000
            omega_max = 0.9
            omega_min = 0.4
            c1 = 2
            c2 = 2

            w = np.matmul(w, w)
            fitness_func = fitness(w, y, G)
            b, MI_array = PSO(fitness_func, part_num, iter_num, omega_max, omega_min, c1, c2)
            endtime = time.time()
            runtime = endtime - starttime
            print('time: {:.2f}'.format(runtime))
            print(b, MI_array[-1])

            df = df.append(pd.DataFrame(np.array([part_num, iter_num, b, MI_array]).reshape(1, -1), columns=['part_num', 'iter_num', 'b', 'MI']))
            plt.plot(MI_array)

    df.to_csv(output_path + \
        'w2_dim_{:d}_samplenum_{:d}_partnum_5102030_noconstrained.csv'.format(shape, sample_num),
        index=False
        )
    plt.savefig(output_path + \
        'w2_dim_{:d}_samplenum_{:d}_partnum_5102030_noconstrained.png'.format(shape, sample_num),
        )
    plt.show()


def main_2():
    NODE_PATH = "D:/code/DTI_data/network_distance/AAL_90_num.node"
    GRAPH_PATH = "D:/code/DTI_data/network_distance/grouplevel.edge"
    output_path = "D:/code/mutual_information_toy_output/"
    shape = 32
    sample_num = 100
    random_seed = 123456
    x, y, _ = nxnetwork_generate(shape, sample_num, random_seed)

    node_df = pd.read_csv(NODE_PATH, sep=' ', header=None)
    region = node_df.iloc[:, 3].to_numpy()
    node1 = np.where(region==1)[0]
    node2 = np.where(region==2)[0]
    node3 = np.where(region==3)[0]
    node4 = np.where(region==4)[0]
    node5 = np.where(region==5)[0]
    
    ggraph = pd.read_csv(GRAPH_PATH, sep='\t', header=None).to_numpy()
    g1 = np.matrix(ggraph[node1, :][:, node1])
    g2 = np.matrix(ggraph[node2, :][:, node2])
    g3 = np.matrix(ggraph[node3, :][:, node3])
    g4 = np.matrix(ggraph[node4, :][:, node4])
    g5 = np.matrix(ggraph[node5, :][:, node5])

    g = g3[:shape, :][:, :shape]
    starttime = time.time()

    # PSO parameters
    part_num = 20
    iter_num = 300
    omega_max = 0.9
    omega_min = 0.4
    c1 = 2
    c2 = 2

    # 10-fold validation
    cv = 10
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_seed)
    acc_sum = 0
    for idx, (train_idx, test_idx) in enumerate(kf.split(x)):
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = MI_learning(x_train, y_train, g, 4)
        b_list, MI_list = model.learning(part_num, iter_num, omega_max, omega_min, c1, c2)
        # f1_train = np.matmul(x_train, b1)
        # f1_test = np.matmul(x_test, b1)
        # df = pd.DataFrame(np.array([part_num, iter_num, b1, MI_array1]).reshape(1, -1), columns=['part_num', 'iter_num', 'b', 'MI'])
        # plt.plot(MI_array1)

        # df.to_csv(output_path + \
        #     'w1_dim_{:d}_part_num_{:d}.csv'.format(shape, part_num),
        #     index=False
        #     )
        # plt.savefig(output_path + \
        #     'w1_dim_{:d}_part_num_{:d}.png'.format(shape, part_num)
        #     )


        # matrix to array
        f1_train = x_train.reshape(x_train.shape[0], -1)
        f1_test = x_test.reshape(x_test.shape[0], -1)

        # Norm
        scaler = StandardScaler()
        scaler.fit(f1_train)
        f1scale_train = scaler.transform(f1_train)
        f1scale_test = scaler.transform(f1_test)

        # PCA
        # pca = PCA(n_components=shape)
        # pca.fit(f1scale_train)
        # f1scale_train = pca.transform(f1scale_train)
        # f1scale_test = pca.transform(f1scale_test)


        # SVC
        svc = SVC(kernel='rbf', random_state=1, gamma=0.001, C=10)
        model = svc.fit(f1scale_train, y_train)

        predict_train = model.predict(f1scale_train)
        correct_train = np.sum(predict_train == y_train)
        accuracy_train = correct_train / train_idx.size

        predict = model.predict(f1scale_test)
        correct = np.sum(predict == y_test)
        accuracy = correct / test_idx.size
        print("cv: {}/{}, acc.: {:.1f}/{:.1f}\n".format(idx, cv, accuracy_train*100, accuracy*100))
        acc_sum += accuracy
        print("total acc.: {:.1f}\n".format(acc_sum / cv * 100))

    endtime = time.time()
    runtime = endtime - starttime


if __name__ == "__main__":
    main_2()