import numpy as np
import matplotlib.pyplot as plt


class MRF_logpdf_unnormalized():
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def __call__(self, value):
        w = self.w
        b = self.b
        # value is a row vector
        return 0.5 * value * w * value.T + b * value.T


def metropolis(func, start, steps=10000):
    samples = np.zeros((steps, start.size))
    old_x = start
    old_prob = func(old_x)
    for i in range(steps):
        new_x = old_x + np.random.normal(0, 5, start.size)
        new_prob = func(new_x)
        acceptance = new_prob - old_prob
        if acceptance >= np.log(np.random.random()):
            samples[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            samples[i] = old_x
    return samples


w = np.matrix(np.random.normal(0, 1, (90, 90)))
b = np.matrix(np.random.normal(0, 1, 90))
start = np.matrix(np.random.normal(0, 1, 90))
mrf = MRF_logpdf_unnormalized(w, b)

trace = metropolis(mrf, start=start, steps=7000000)
chain = trace[1000000:]

_, ax = plt.subplots(nrows=5, ncols=2)

for i in range(5):
    print(chain[:,i])
    ax[i,0].hist(chain[:,i], bins=50, density=True, label="x_{}".format(i))
    ax[i,0].legend(fontsize=12)
    ax[i,1].plot(chain[:,i])
ax[4,0].set_xlabel('$x$', fontsize=16)
ax[2,0].set_ylabel('$pdf(x)$', fontsize=16)
plt.tight_layout()
plt.show()
