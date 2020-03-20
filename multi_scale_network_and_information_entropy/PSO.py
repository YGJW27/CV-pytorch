import time
import numpy as np
import matplotlib.pyplot as plt

from ggmfit import *
from mutual_information import *
from sphere import *
from data_loader import *

class fitness():
    def __init__(self, w, y, G):
        self.w = w
        self.y = y
        self.G = G
        self.dim = w.shape[1]

    def __call__(self, b):
        f = np.matmul(self.w, b)
        MI = mutual_information(f, self.y, self.G)
        return MI


def velocity(v, pbest_pos, gbest_pos, p_n, x, omega, c1, c2):
    v = omega * v + \
        c1 * np.random.uniform(size=(p_n, 1)) * (pbest_pos - x) + \
        c2 * np.random.uniform(size=(p_n, 1)) * (gbest_pos.reshape(1, -1).repeat(p_n, axis=0) - x)

    # velocity limit
    v[v >= 0.1] = 0.1
    v[v <= -0.1] = -0.1
    return v


def PSO(fitness, part_num, iter_num, omega_max, omega_min, c1, c2):
    dim = fitness.dim
    x = np.random.uniform(-1, 1, size=(part_num, dim))
    x = x / np.linalg.norm(x, axis=1).reshape(-1, 1)
    v = np.random.uniform(size=(part_num, dim))
    fit = np.zeros(part_num)
    for i in range(part_num):
        fit[i] = fitness(x[i])
    pbest_pos = x
    pbest_fit = fit
    gbest_pos = x[np.argmax(fit)]
    gbest_fit = np.max(fit)
    fit_best_list = [gbest_fit]

    for it in range(iter_num):
        omega = omega_max - (omega_max - omega_min) * it / iter_num
        v = velocity(v, pbest_pos, gbest_pos, part_num, x, omega, c1, c2)
        x = x + v
        x = x / np.linalg.norm(x, axis=1).reshape(-1, 1)

        fit_next = np.zeros(part_num)
        for i in range(part_num):
            fit_next[i] = fitness(x[i])

        pbest_pos[fit_next > fit] = x[fit_next > fit]
        fit[fit_next > fit] = fit_next[fit_next > fit]
        gbest_pos = x[np.argmax(fit)]
        gbest_fit = np.max(fit)
        fit_best_list.append(gbest_fit)

    return gbest_pos, np.array(fit_best_list)


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
    fitness_func = fitness(w, y, G)
    b, MI_array = PSO(fitness_func, part_num, iter_num, omega_max, omega_min, c1, c2)
    endtime = time.time()
    runtime = endtime - starttime
    print('time: {:.2f}'.format(runtime))
    print(b)
    plt.plot(MI_array)
    plt.show()

if __name__ == "__main__":
    main()