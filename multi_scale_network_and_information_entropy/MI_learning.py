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
            fitfunc = fitness(x, y, g)
            self.funcs.append(fitfunc)
            x = np.matmul(x, self.x)

    def learning(self, part_num, iter_num, omega_max, omega_min, c1, c2):
        b_list = []
        MI_list = []
        for fitfunc in self.funcs:
            b, MI_array = PSO(fitfunc, part_num, iter_num, omega_max, omega_min, c1, c2)
            b_list.append(b)
            MI_list.append(MI_array)

        return b_list, MI_list