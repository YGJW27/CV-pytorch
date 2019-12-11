import numpy as np
import matplotlib.pyplot as plt


class MRF_logpdf_unnormalized():
    def __init__(self, w):
        self.w = w

    def __call__(self, value):
        w = self.w
        if value > 0:
            return np.log(w) - w * value
        else:
            return -100


def metropolis(func, start, steps=10000):
    samples = np.zeros(steps)
    old_x = start
    old_prob = func(old_x)
    for i in range(steps):
        new_x = old_x + np.random.normal(0, 0.5)
        new_prob = func(new_x)
        acceptance = new_prob - old_prob
        if acceptance >= np.log(np.random.random()):
            samples[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            samples[i] = old_x
    return samples


w = 1
start = 1
mrf = MRF_logpdf_unnormalized(w)
x = np.linspace(0, 10, 100)
y = np.exp(np.log(w) - w * x)
plt.xlim(0, 8)
plt.plot(x, y, 'r-')
trace = metropolis(mrf, start=start, steps=10000)
chain = trace[1000:]


plt.hist(chain, bins=100, density=True)
plt.show()
