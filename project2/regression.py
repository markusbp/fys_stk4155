import numpy as np
import math
import scipy

from sklearn import linear_model
from numba import jit

from tqdm import trange

class Linear:
    # Class for ordinary least squares linear regression
    # mimic sklearn syntax to use both interchangeably
    def __init__(self, p):
        self.p = p # degree of polynomial
        self.combinations = int(0.5*(self.p+1)*(self.p+2)) # number of combinations of x,y

    #@jit # speed up, somewhat
    def design_matrix(self, r):
        # create design matrix for polynomial regression
        x = np.zeros((len(r), self.combinations)) # design matrix
        count = 0
        for i in range(self.p + 1): # 0, 1, ... p
            for j in range(self.p + 1 - i): # p, p-1, ... 0
                x[:, count] = r[:, 0]**j*r[:, 1]**i # unique powers of x,y
                count += 1
        return x

    def fit(self, x, y):
        d = self.design_matrix(x)
        self.beta_ = np.linalg.pinv(d)@y # least squares estimator

    def predict(self, data):
        d = self.design_matrix(data)
        return d@self.beta_ # y ~ XB

    def gradient(self, x, y, beta, n):
        # gradient of error wrt. beta
        d = self.design_matrix(x)
        return 2/n*d.T@(d@beta - y)

    def fit_sgd(self, x, y, bs, epochs, trainer):
        '''
        Fit regression model using stochastic gradient descent
        x : input
        y : targets/labels
        bs : minibatch size
        epochs: Number of time to run through dataset
        lr : learning rate
        mom: momentum
        '''
        beta = np.random.uniform(-0.01, 0.01, self.combinations) # initial guess IMPROVE
        v = 0 # zero moemntum to start
        steps = len(x)//bs

        for i in trange(epochs, desc = 'Training'):
            shuffle_inds = np.random.choice(len(x), len(x), replace = False)
            x_train = x[shuffle_inds] # shuffle dataset
            y_train = y[shuffle_inds]

            for j in range(steps):
                batch_inds = np.random.choice(len(x), bs)
                batch_x = x_train[batch_inds] # random minibatches
                batch_y = y_train[batch_inds]
                grad = self.gradient(batch_x, batch_y, beta, bs)
                beta = trainer.step(beta, grad, i) # gradient descent step
            self.beta_ = beta

class RidgeRegression(Linear):
    # Ridge regression, inherit from Linear class
    def __init__(self, p, lam):
        super().__init__(p)
        self.lam = lam # shrinkage parameter

    def fit(self, x, y):
        d = self.design_matrix(x) # inherited
        self.beta_ = np.linalg.inv(d.T@d + self.lam*np.eye(d.shape[-1]))@d.T@y

    def gradient(self, x, y, beta, n):
        # gradient of the error wrt. beta
        d = self.design_matrix(x)
        return 2/n*d.T@(d@beta - y) - 2*beta
