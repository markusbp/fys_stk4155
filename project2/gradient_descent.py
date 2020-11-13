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
        self.v_ = 0 # initial value for momentum term/memory
        self.decay = decay # whether to apply learning rate decay, bool

    def step(self, x, grad, epoch = 1, d = 1):
        # perform gradient descent step
        # x: quantity to update, grad: its gradient, d: number of batches
        self.v_ = self.momentum*self.v_ + self.lr(epoch, d)*grad
        return x - self.v_

    def lr(self, epoch, n_batches = 1):
        # learning rate with/without decay
        if self.decay:
            return self.lr0/(epoch*n_batches + 1)
        else:
            return self.lr0
