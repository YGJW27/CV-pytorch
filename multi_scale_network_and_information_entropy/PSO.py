import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


def velocity_gradient(v, fitness, gbest_pos, p_n, x, omega, c1, c2):
    gradient = np.zeros((p_n, fitness.dim))
    for i in range(p_n):
        f = np.matmul(fitness.w, x[i])
        gradient[i] = mutual_information_multinormal_gradient(f, fitness.y, fitness.w, x[i])
    v = omega * v + \
        c1 * np.random.uniform(size=(p_n, 1)) * gradient + \
        c2 * np.random.uniform(size=(p_n, 1)) * (gbest_pos.reshape(1, -1).repeat(p_n, axis=0) - x)

    # velocity limit
    v[v >= 0.1] = 0.1
    v[v <= -0.1] = -0.1
    return v


def PSO(fitness, part_num, iter_num, omega_max, omega_min, c1, c2):
    dim = fitness.dim
    x = np.random.uniform(-1, 1, size=(part_num, dim))
    x = x / np.linalg.norm(x, axis=1).reshape(-1, 1)
    v = np.random.uniform(-0.1, 0.1, size=(part_num, dim))
    fit_list = []

    fit = np.zeros(part_num)
    for i in range(part_num):
        fit[i] = fitness(x[i])
    fit_list.append(fit)
    pbest_pos = x
    pbest_fit = fit
    gbest_pos = x[np.argmax(fit)]
    gbest_fit = np.max(fit)
    fit_best_list = [gbest_fit]
    b_track = [gbest_pos]

    for it in range(iter_num):
        omega = omega_max - (omega_max - omega_min) * it / iter_num
        v = velocity(v, pbest_pos, gbest_pos, part_num, x, omega, c1, c2)
        # v = velocity_gradient(v, fitness, gbest_pos, part_num, x, omega, c1, c2) # gradient based
        x = x + v
        x = x / np.linalg.norm(x, axis=1).reshape(-1, 1)

        fit_next = np.zeros(part_num)
        for i in range(part_num):
            fit_next[i] = fitness(x[i])
        fit_list.append(fit_next)
        pbest_pos[fit_next > fit] = x[fit_next > fit]
        fit[fit_next > fit] = fit_next[fit_next > fit]
        gbest_pos = x[np.argmax(fit)]
        gbest_fit = np.max(fit)
        fit_best_list.append(gbest_fit)
        b_track.append(gbest_pos)

        convergence_particle = np.sum(np.linalg.norm(
            pbest_pos - gbest_pos.reshape(1, -1).repeat(part_num, axis=0),
            axis=1) < 10e-5)
        if convergence_particle >= 0.6 * part_num:
            break

        if it % 10 == 0:
            print("iter: {:d}, \t convergence particle: {:d}\n".format(
                it, convergence_particle))

    return gbest_pos, np.array(fit_best_list), np.array(fit_list).T, np.array(b_track)


def main():
    output_path = 'D:/code/mutual_information_toy_output/'
    shape = 4
    sample_num = 1000
    random_seed = 123456
    df = pd.DataFrame(columns=['iter_num', 'b', 'MI'])
    for randi in range(10):
        w, y, _ = graph_generate(shape, sample_num, random_seed+randi)

        G = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        G = np.ones((shape, shape))

        starttime = time.time()
        part_num = 20
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
        print(b, MI_array[-1])

        df = df.append(pd.DataFrame(np.array([iter_num, b, MI_array]).reshape(1, -1), columns=['iter_num', 'b', 'MI']))
        
        plt.plot(MI_array)
    df.to_csv(output_path + \
        'dim_{:d}_samplenum_{:d}_partnum_{:d}_10times_noconstrained_grad.csv'.format(shape, sample_num, part_num),
        index=False
        )
    plt.savefig(output_path + \
        'dim_{:d}_samplenum_{:d}_partnum_{:d}_10times_noconstrained_grad.png'.format(shape, sample_num, part_num),
        )
    plt.show()




if __name__ == "__main__":
    main()