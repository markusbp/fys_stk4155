import numpy as np

class MSE(object):
    def __call__(self, x, y):
        return np.mean((x-y)**2)

    def gradient(self, x, y):
        return np.zeros(x.shape)
