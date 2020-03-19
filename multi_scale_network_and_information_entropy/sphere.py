import numpy as np


def sphere(sample_num=1000):
    coordinates = np.zeros((sample_num, 3))
    for i in range(sample_num):
        phi = np.random.uniform(0, 1) * np.pi * 2
        theta = np.random.uniform(0, 1) * np.pi
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        coordinates[i] = [x, y, z]
    return coordinates


def super_sphere(dim, sample_num=1000):
    coordinates = np.zeros((sample_num, dim))
    for i in range(sample_num):
        coord = np.random.uniform(-1, 1, dim)
        coordinates[i] = coord / np.linalg.norm(coord)
    return coordinates


if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import matplotlib.pyplot as plt

    coo = super_sphere(3, 1000)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(coo[:, 0], coo[:, 1], coo[:, 2])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
