import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

b = np.array([
    [-0.02008246, 0.62588954, 0.77965311],
    [0.01284789, 0.58548699, 0.81057999],
    [-0.0140248, 0.55161179, 0.83398306],
    [0.02004923, 0.61888339, 0.78522696],
    [0.03365703, 0.56455624, 0.82470811],
    [0.01068077, 0.5724645, 0.81985993],
    [-0.001588, 0.59407464, 0.80440836],
    [0.02433776, 0.61714206, 0.78647527],
    [-0.01379456, 0.59008708, 0.80722175],
    [-0.03937142, 0.58673142, 0.80882392],
    [0.00027985, 0.63891008, 0.76928137]
    ])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(b[:, 0], b[:, 1], b[:, 2], c='r')
# ax.plot_trisurf(b[:, 0], b[:, 1], b[:, 2], c='r')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.xlim(-1 * 1.1, 1 * 1.1)
plt.ylim(-1 * 1.1, 1 * 1.1)
ax.set_zlim(-1 * 1.1, 1 * 1.1)
plt.savefig("D:/" + 'fig2.png')
plt.show()