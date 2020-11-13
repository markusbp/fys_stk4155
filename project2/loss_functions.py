import numpy as np
# module containing all loss functions
# For all classes: x: model output, y: target

class MSE(object):
    def __call__(self, x, y):
        return np.mean((x-y)**2)

    def gradient(self, x, y):
        return -2*(y - x)

class BinaryCrossEntropy(object): # never used :(
    def __call__(self, x, y):
        loss = y*np.log(x) + (1-y)*np.log(1-x)
        return -np.mean(loss)

class CategoricalCrossEntropy(object): # works with e.g. sigmoid, not softmax
    def __call__(self, x, y):
        return -np.mean(y*np.log(x))*y.shape[-1] # n_classes

    def gradient(self, x, y):
        return -y/x

class SoftmaxCrossEntropy(object):
    '''
        Loss function used for classification tasks;
        correctly handles derivative of softmax
    '''
    def __call__(self, x, y):
        return -np.mean(y*np.log(x))*y.shape[-1] # don't mean over classes

    def gradient(self, x, y):
        return x - y

    def output_error(self, z, a, grad_z, y):
        # correct softmax output "error"/gamma
        return a - y
