import numpy as np

class GD(object):
    def __init__(self, lr0, momentum = 0, decay = False):
        '''
        Gradient Descent class
        lr0: initial learning rate
        momentum: momentum parameter
        '''
        self.lr0 = lr0
        self.momentum = momentum
        self.v_ = 0
        self.decay = decay

    def step(self, x, grad, epoch = 1, d = 1):
        self.v_ = self.momentum*self.v_ + self.lr(epoch, d)*grad
        return x - self.v_

    def reset(self):
        self.v_ = 0

    def lr(self, epoch, n_batches = 1):
        if self.decay:
            return self.lr0/(epoch*n_batches + 1)
        else:
            return self.lr0
