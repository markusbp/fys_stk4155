import numpy as np

class MSE(object):
    def __call__(self, x, y):
        return np.mean((x-y)**2)

    def gradient(self, x, y):
        return -2*(y - x)

class BinaryCrossEntropy(object):
    def __call__(self, x, y):
        loss = y*np.log(x) + (1-y)*np.log(1-x)
        return -np.mean(loss)

class CategoricalCrossEntropy(object):
    def __call__(self, x, y):
        return -np.mean(y*np.log(x))*y.shape[-1] # n_classes

    def gradient(self, x, y):
        return -y/x


class SoftmaxCrossEntropy(object):
    def __call__(self, x, y):
        return -np.mean(y*np.log(x))*y.shape[-1]

    def gradient(self, x, y):
        return x - y

    def output_error(self, z, a, grad_z, y):
        return a - y
