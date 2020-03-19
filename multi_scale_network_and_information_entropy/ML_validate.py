import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from ggmfit import *
from mutual_information import *
from sphere import *

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from ggmfit import *
from mutual_information import *
from sphere import *




# np.random.seed(11233)

# x0_mu = np.array([15, 7, 12, 8])
# x0_cov = np.array([[3, 0.2, 0.5, 0.3], [0.2, 1, 0, 0.1], [0.5, 0, 2, 0.5], [0.3, 0.1, 0.5, 2]])
# # x0_cov = np.array([[1, 0.2, 0.5], [0.2, 1, 0], [0.5, 0, 1]])
# x1_mu = np.array([13, 8, 11, 10])
# x1_cov = np.array([[1, 0.6, 1, 0.2], [0.6, 0.6, 0, 0.1], [1, 0, 3, 0.3], [0.2, 0.1, 0.3, 3]])
# # x1_cov = np.array([[1, 0.6, 0.9], [0.6, 0.6, 0], [0.9, 0, 3]])

# x0 = np.random.multivariate_normal(x0_mu, x0_cov, 2000)
# x1 = np.random.multivariate_normal(x1_mu, x1_cov, 2000)
# y0 = np.zeros(500)
# y1 = np.ones(500)
# x = np.concatenate((x0, x1), axis=0)
# x = np.reshape(x, (-1, 4, 4))
# y = np.concatenate((y0, y1))

# w = np.zeros(shape=(x.shape[0], x.shape[1], x.shape[1]))
# for i, wi in enumerate(w):
#     # rand0_1 = np.random.uniform()
#     # w[i][0] = np.array([0, x[i][0] * rand0_1, x[i][0] * (1 - rand0_1)])
#     # rand0_1 = np.random.uniform()
#     # w[i][1] = np.array([x[i][1] * rand0_1, 0, x[i][1] * (1 - rand0_1)])
#     # rand0_1 = np.random.uniform()
#     # w[i][2] = np.array([x[i][2] * rand0_1, x[i][2] * (1 - rand0_1), 0])
#     w[i] = (x[i] + x[i].T) / 2
#     np.fill_diagonal(w[i], 0)

# # w = np.matmul(w, w)


# # G = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
# G = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

# array_num = 1000
# b_spots = super_sphere(4, array_num)
# MI_array = np.zeros(array_num)
# for i in range(1000):
#     f = np.matmul(w, b_spots[i])
#     MI_array[i] = mutual_information(f, y, G)
#     MI_2 = mutual_information(2*f, y, G)
#     MI = mutual_information_multinormal(f, y)
#     grad = mutual_information_multinormal_gradient(f, y, w, b_spots[i])

# i_max = np.argmax(MI_array)
# b_max = b_spots[i_max]
# MI_max = MI_array[i_max]
# grad_max = mutual_information_multinormal_gradient(f, y, w, b_max)
# print('b: ', b_max, '\nMI: ', MI_max, '\ngrad: ',grad_max)


np.random.seed(13)
x0_mu = np.array([15, 7, 12, 8])
x0_cov = np.array([[3, 0.2, 0.5, 0.3], [0.2, 1, 0, 0.1], [0.5, 0, 2, 0.5], [0.3, 0.1, 0.5, 2]])
# x0_cov = np.array([[1, 0.2, 0.5], [0.2, 1, 0], [0.5, 0, 1]])
x1_mu = np.array([13, 8, 11, 10])
x1_cov = np.array([[1, 0.6, 1, 0.2], [0.6, 0.6, 0, 0.1], [1, 0, 3, 0.3], [0.2, 0.1, 0.3, 3]])
# x1_cov = np.array([[1, 0.6, 0.9], [0.6, 0.6, 0], [0.9, 0, 3]])

x0 = np.random.multivariate_normal(x0_mu, x0_cov, 200)
x1 = np.random.multivariate_normal(x1_mu, x1_cov, 200)
y0 = np.zeros(50)
y1 = np.ones(50)
x = np.concatenate((x0, x1), axis=0)
x = np.reshape(x, (-1, 4, 4))
y = np.concatenate((y0, y1))

w = np.zeros(shape=(x.shape[0], x.shape[1], x.shape[1]))
for i, wi in enumerate(w):
    # rand0_1 = np.random.uniform()
    # w[i][0] = np.array([0, x[i][0] * rand0_1, x[i][0] * (1 - rand0_1)])
    # rand0_1 = np.random.uniform()
    # w[i][1] = np.array([x[i][1] * rand0_1, 0, x[i][1] * (1 - rand0_1)])
    # rand0_1 = np.random.uniform()
    # w[i][2] = np.array([x[i][2] * rand0_1, x[i][2] * (1 - rand0_1), 0])
    w[i] = (x[i] + x[i].T) / 2
    np.fill_diagonal(w[i], 0)


x_all_train = np.reshape(np.concatenate((w[0:40], w[50:90]), axis=0), (80, -1))
x_all_test = np.reshape(np.concatenate((w[40:50], w[90:100]), axis=0), (20, -1))

b = np.array([0.22008872, -0.95572638, 1, 0.04957334])
f = np.matmul(w, b)
x_MI_train = np.concatenate((f[0:40], f[50:90]), axis=0)
x_MI_test = np.concatenate((f[40:50], f[90:100]), axis=0)

y_train = np.concatenate((y[0:40], y[50:90]))
y_test = np.concatenate((y[40:50], y[90:100]))

# all_Norm
scaler = StandardScaler()
scaler.fit(x_all_train)
x_all_train = scaler.transform(x_all_train)
x_all_test = scaler.transform(x_all_test)

# MI_Norm
scaler = StandardScaler()
scaler.fit(x_MI_train)
x_MI_train = scaler.transform(x_MI_train)
x_MI_test = scaler.transform(x_MI_test)

# PCA
pca = PCA(n_components=3)
pca.fit(x_all_train)
x_allpca_train = pca.transform(x_all_train)
x_allpca_test = pca.transform(x_all_test)

# SVC
svc = SVC(kernel='rbf', random_state=1, gamma=0.5, C=1)
model = svc.fit(x_MI_train, y_train)

predict_train = model.predict(x_MI_train)
correct_train = np.sum(predict_train == y_train)
accuracy_train = correct_train / 80

predict = model.predict(x_MI_test)
correct = np.sum(predict == y_test)
accuracy = correct / 20

print(accuracy_train, accuracy)