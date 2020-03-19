# import torch
import time
import numpy as np
import matplotlib.pyplot as plt

from ggmfit import *
from mutual_information import *
from sphere import *
from data_loader import *

def fitness(w, y, G, b):
    f = np.matmul(w, b)
    MI = mutual_information(f, y, G)
    return MI


def velocity(v, pbest_pos, gbest_pos, p_n, b, omega, c1, c2):
    v = omega * v + \
        c1 * np.random.uniform(size=(p_n, 1)) * (pbest_pos - b) + \
        c2 * np.random.uniform(size=(p_n, 1)) * (gbest_pos.reshape(1, -1).repeat(p_n, axis=0) - b)
    return v


def PSO(w, y, G, part_num, iter_num, omega_max, omega_min, c1, c2):
    b = np.random.uniform(-1, 1, size=(part_num, w.shape[1]))
    b = b / np.linalg.norm(b, axis=1).reshape(-1, 1)
    v = np.random.uniform(size=(part_num, w.shape[1]))
    MI = np.zeros(part_num)
    for i in range(part_num):
        MI[i] = fitness(w, y, G, b[i])
    pbest_pos = b
    pbest_MI = MI
    gbest_pos = b[np.argmax(MI)]
    gbest_MI = np.max(MI)
    MI_best_list = [gbest_MI]

    for it in range(iter_num):
        omega = omega_max - (omega_max - omega_min) * it / iter_num
        v = velocity(v, pbest_pos, gbest_pos, part_num, b, omega, c1, c2)
        b = b + v
        b = b / np.linalg.norm(b, axis=1).reshape(-1, 1)

        MI_next = np.zeros(part_num)
        for i in range(part_num):
            MI_next[i] = fitness(w, y, G, b[i])

        pbest_pos[MI_next > MI] = b[MI_next > MI]
        MI[MI_next > MI] = MI_next[MI_next > MI]
        gbest_pos = b[np.argmax(MI)]
        gbest_MI = np.max(MI)
        MI_best_list.append(gbest_MI)

    return gbest_pos, np.array(MI_best_list)


def main():
    shape = 5
    sample_num = 500
    random_seed = 3224107
    w, y, _ = graph_generate(shape, sample_num, random_seed)

    G = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    G = np.ones((shape, shape))

    starttime = time.time()
    part_num = 10
    iter_num = 150
    omega_max = 0.9
    omega_min = 0.4
    c1 = 2
    c2 = 2
    b, MI_array = PSO(w, y, G, part_num, iter_num, omega_max, omega_min, c1, c2)
    endtime = time.time()
    runtime = endtime - starttime
    print('time: {:.2f}'.format(runtime))
    print(b)
    plt.plot(MI_array)
    plt.show()

if __name__ == "__main__":
    main()