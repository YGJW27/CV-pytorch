import time
import numpy as np
from PSO import *

class MI_learning():
    def __init__(self, x, y, g, k):
        '''
        k is the number of feature layer
        '''
        self.funcs = []
        self.x = x
        for i in range(k):
            scale = np.mean(x[x != 0])
            fitfunc = fitness(x/scale, y, g)
            self.funcs.append(fitfunc)
            x = np.matmul(x, self.x)

    def learning(self, part_num, iter_num, omega_max, omega_min, c1, c2):
        b_list = []
        MI_list = []
        b_track_list = []
        for fitfunc in self.funcs:
            learn_start = time.time()
            b, MI_array, _, b_track  = PSO(fitfunc, part_num, iter_num, omega_max, omega_min, c1, c2)
            learn_complete = time.time()
            learn_time = learn_complete - learn_start
            print('learn time:', learn_time, '\n')
            b_list.append(b)
            MI_list.append(MI_array)
            b_track_list.append(b_track)

        return b_list, MI_list, b_track_list