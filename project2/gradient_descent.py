import numpy as np

class GD(object):
    def __init__(self, lr0, momentum):
        '''
        Gradient Descent class
        lr0: initial learning rate
        momentum: momentum parameter
        '''
        self.lr0 = lr0
        self.momentum = momentum
        self.v_ = 0

    def step(self, x, grad, epoch = 1):
        self.v_ = self.momentum*self.v_ + self.lr(epoch)*grad
        return x - self.v_

    def reset(self):
        self.v_ = 0

    def lr(self, epoch, d = 0.1):
        return self.lr0/(epoch*d + 1)
