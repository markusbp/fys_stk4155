import numpy as np

class Sigmoid(object):
    def __call__(self, x):
        return 1/(1 + np.exp(-x))

    def gradient(self, x):
        return self(x)**2*np.exp(-x)

class Relu(object):
    def __call__(self, x):
        # apparently in-place maxmimum is faster, hence out = x
        return np.maximum(0, x, out = x)

    def gradient(self, x):
        return np.where(x > 0, 1, 0)

class LeakyRelu(object):
    def __call__(self, x):
        return np.where(x > 0, x, 0.01*x)

    def gradient(self, x):
        return np.where( x > 0, 1, 0.01)

class Softmax(object):
    def __call__(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis = -1, keepdims = True)
