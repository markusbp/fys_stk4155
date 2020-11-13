import numpy as np
# module containing activation functions, hopefully self-explanatory!

class Sigmoid(object):
    def __call__(self, x):
        return 1/(1 + np.exp(-x))

    def gradient(self, x):
        s = self(x)
        return s*(1 - s)

class Relu(object):
    def __call__(self, x):
        return np.maximum(0.0, x)

    def gradient(self, x):
        return np.where(x > 0.0, 1.0, 0.0)

class LeakyRelu(object):
    def __init__(self, a = 0.01):
        self.a = a

    def __call__(self, x):
        return np.where(x > 0.0, x, self.a*x)

    def gradient(self, x):
        return np.where(x > 0.0, 1.0, self.a)

class Softmax(object):
    def __call__(self, x):
        z = np.exp(x)
        return z/np.sum(z, axis = -1, keepdims = True)

    def gradient(self, x):
        s = self(x)
        return s - s**2

class Linear(object):
    def __call__(self, x):
        return x

    def gradient(self, x):
        return np.ones_like(x)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # visualize all activation functions
    n = 100
    x = np.linspace(-5, 5, n)
    funcs = [Sigmoid(), Relu(), LeakyRelu(0.05), Linear()]
    names = ['Sigmoid', 'Relu', 'Leaky Relu', 'Linear']
    fig, axs = plt.subplots(2,2)
    row = [0, 0, 1, 1]
    col = [0, 1, 0, 1]
    for func, name, r, c in zip(funcs, names, row, col):
        y = func(x)
        y_prime = func.gradient(x)

        if name == 'Relu' or name == 'Leaky Relu':
            y_prime[n//2] = None
        axs[r, c].plot(x, y, label = 'Function', linewidth = 0.75)
        axs[r, c].plot(x, y_prime,  label = 'Derivative', linewidth = 0.75)
        axs[r, c].legend(frameon = False)
        axs[r, c].set_title(name)
        axs[r, c].set_xlabel('x', fontsize = 12)

    plt.tight_layout()
    plt.savefig('./results/activations.png')
