import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from ggmfit import *
from mutual_information import *
from sphere import *
from data_loader import *

def b_learning(w, y, lr=0.05, momentum=0.9, epoch=1000, init_random=0):
    np.random.seed(init_random)
    b = np.random.uniform(-1, 1, size=w.shape[1])
    b = b / np.linalg.norm(b)
    MI_array = np.zeros(epoch)
    v = np.zeros_like(b)

    # w = np.matmul(w, w)

    for i in range(epoch):
        f = np.matmul(w, b)
        MI_array[i] = mutual_information_multinormal(f, y)
        b_grad = mutual_information_multinormal_gradient(f, y, w, b)
        v = momentum * v + b_grad
        # b = b + lr * v
        b = b + v * (lr ** (1 - i / epoch))
        b = b / np.linalg.norm(b)
    return b, MI_array, b_grad

shape = 5
sample_num = 500
random_seed = 3224107
w, y, _ = graph_generate(shape, sample_num, random_seed)

G = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
G = np.ones((shape, shape))

# for rand in range(10):
#     b, MI_array, grad = b_learning(w, y, 0.01, 0.9, 1000, rand)
#     print('\n', rand, ': ', b, grad)
#     plt.plot(MI_array)
# plt.show()

"""
array_num = 10000
b_spots = super_sphere(shape, array_num)
MI_array = np.zeros(array_num)
for i in range(array_num):
    f = np.matmul(w, b_spots[i])
    MI_array[i] = mutual_information(f, y, G)
    MI_2 = mutual_information(2*f, y, G)
    MI = mutual_information_multinormal(f, y)
    grad = mutual_information_multinormal_gradient(f, y, w, b_spots[i])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(b_spots[:, 0], b_spots[:, 1], b_spots[:, 2], c=MI_array, cmap='Blues')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.savefig("D:/" + 'fig3.png')
plt.show()
"""

def convex_test(w, y, num=10000):
    bigger = np.zeros(num)
    for i in range(num):
        b1 = np.random.uniform(-1, 1, size=w.shape[1])
        b1 = b1 / np.linalg.norm(b1)
        f1 = np.matmul(w, b1)
        MI1 = mutual_information_multinormal(f1, y)

        b2 = np.random.uniform(-1, 1, size=w.shape[1])
        b2 = b2 / np.linalg.norm(b2)
        f2 = np.matmul(w, b2)
        MI2 = mutual_information_multinormal(f2, y)

        b12 = b1 + b2
        b12 = b12 / np.linalg.norm(b12)
        f12 = np.matmul(w, b12)
        MI12 = mutual_information_multinormal(f12, y)

        bigger[i] = (MI12 >= (MI1 + MI2) / 2)

    return bigger

bigger = convex_test(w, y)