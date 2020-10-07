import numpy as np
import math
import scipy

from sklearn import linear_model
from numba import jit

class Linear:
    # Class for ordinary least squares linear regression
    # mimic sklearn syntax to use both interchangeably
    def __init__(self, p):
        self.p = p # degree of polynomial

    @jit # speed up, somewhat
    def design_matrix(self, r):
        # create design matrix for polynomial regression
        combinations = int(0.5*(self.p+1)*(self.p+2)) # number of combinations of x,y
        x = np.zeros((len(r), combinations)) # design matrix
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

class RidgeRegression(Linear):
    # Ridge regression, inherit from Linear class
    def __init__(self, p, lam):
        super().__init__(p)
        self.lam = lam # shrinkage parameter

    def fit(self, x, y):
        d = self.design_matrix(x) # inherited
        self.beta_ = np.linalg.inv(d.T@d + self.lam*np.eye(d.shape[-1]))@d.T@y

class LassoRegression(Linear):
    # Lasso regression, using sklearn algorithms, inherit from Linear
    def __init__(self, p, lam):
        super().__init__(p)
        self.lam = lam # shrinkage parameter
        # create sklearn model, fit_intercept false as we add it in design_matrix
        self.model = linear_model.Lasso(self.lam, fit_intercept = False)

    def fit(self, x, y):
        d = self.design_matrix(x) # inherited
        self.model.fit(d, y) # train
        self.beta_ = self.model.coef_ # set beta_ if needed

    def predict(self, data):
        d = self.design_matrix(data)
        return self.model.predict(d)
