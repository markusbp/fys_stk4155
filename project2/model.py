import numpy as np
from tqdm import trange

from gradient_descent import GD

class FFNN(object):
    '''
    Feed Forward Neural Network class
    '''
    def __init__(self, loss):
        self.loss = loss # loss function, as defined in loss_functions.py
        self.layers = []

    def __call__(self, x):
        # compute output of entire model
        for l, layer in enumerate(self.layers):
            x = layer(x) # output is input to next layer
        return x

    def test(self, x, y, metric):
        '''
            Compute model performance for a given metric

            x: input data
            y: target data
            metric: desired metric, must be 'loss', in which case the model loss
            is computed, 'accuracy' (for classification tasks), or 'R2', which
            computes R2 score.
        '''
        pred = self(x) # prediction

        if metric == 'loss':
            result = self.loss(pred, y)
        elif metric == 'accuracy':
            # convert maximum prediction to one_hot vectors
            inds = np.argmax(pred, axis = -1)
            one_hot = np.zeros(pred.shape)
            for i in range(pred.shape[0]):
                one_hot[i, inds[i]] = 1
            # gives +1 only when y and x are maxmimal (1) at the same time
            result = np.mean(np.sum(one_hot*y, axis = -1)) # assume one-hot y
        elif metric == 'r2':
            pass
        return result

    def train(self, x, y, lr, mom = 0, epochs = 10, bs = 100, val_data = None,
              metrics = ['loss'], decay = False):
        '''
            Train FFNN.

            x: input data
            y: target data
            lr: learning rate
            mom: momentum
            epochs: number of epochs
            bs: batch size
            val_data: tuple containing (validation x, validation y) for testing
            metrics: list containing string of desired metrics to compute.
                     Must be 'loss', 'accuracy' or 'R2'.
            decay: Whether to apply learning rate decay during training
        '''
        train_results = np.zeros((epochs, len(metrics))) # save errors
        test_results = np.zeros((epochs, len(metrics)))

        steps = len(x)//bs # number of steps in epoch
        mean_axes = tuple(range(len(x.shape) - 1)) # take mean of gradient over first axes

        n = len(self.layers)
        # set up gradient descent optimizers for each layer, for bias and weigths
        trainer = np.array([[GD(lr, mom, decay), GD(lr, mom, decay)] for i in range(n)])
        train_loop = trange(epochs, desc = 'Training')
        # main train loop
        for i in train_loop:
            for j in range(steps):
                batch_inds = np.random.choice(len(x), bs)
                batch_x = x[batch_inds] # random minibatches
                batch_y = y[batch_inds]
                grad_w, grad_b = self.backprop(batch_x, batch_y)
                # compute gradients, update weights using trainers
                for l, layer in enumerate(self.layers):
                    mean_grad_w = np.mean(grad_w[l], axis = mean_axes) + layer.kernel_reg()
                    mean_grad_b = np.mean(grad_b[l], axis = mean_axes) + layer.bias_reg()
                    layer.kernel = trainer[l,0].step(layer.kernel, mean_grad_w, i, steps)
                    layer.bias = trainer[l,1].step(layer.bias, mean_grad_b, i, steps)

            # compute test error, update progress bare
            description = ''
            for m, metric in enumerate(metrics):
                train_results[i, m] = self.test(x, y, metric)
                if val_data is not None:
                    test_results[i, m] = self.test(val_data[0], val_data[1], metric)
                    description += f'Epoch test {metric}: {test_results[i,m]}. '
            train_loop.desc = description
        return train_results, test_results


    def backprop(self, x, y):
        # backpropagation step, x: inputs, y: targets
        n = len(self.layers) # number of layers

        a = [0 for i in range(n+1)] # activations
        grad = [0 for i in range(n)] # derivative of activation function

        grad_w = [0 for i in range(n)] # gradient of parameters
        grad_b = [0 for i in range(n)]

        a[0] = x # input layer --> activations are just inputs
        # forward pass
        for l, layer in enumerate(self.layers):
            z = layer.forward(x) # compute activities
            grad[l] = layer.activation.gradient(z) # activation function gradient
            x = layer.activation(z) # output is input to next step
            a[l+1] = x # save activations

        # backward pass

        # two cases: one where there might be dependencies between output nodes
        # or the more straight forward case.
        if y.shape[-1] > 1:
            g = self.loss.output_error(z, a[-1], grad[-1], y) # output "error", gamma
        else:
            g = self.loss.gradient(a[-1], y)*grad[-1]

        # backpropagate "error"
        for l in range(1, len(self.layers) + 1):
            if l == 1:
                pass # g already computed for last layer
            else:
                # compute g with g of last layer
                g = np.tensordot(g, self.layers[-l+1].kernel.T, axes = [-1, 1])*grad[-l]
            # save gradients
            grad_w[-l] = np.expand_dims(g, axis = -1)*np.expand_dims(a[-l-1], axis = -2)
            grad_b[-l] = g
        return grad_w, grad_b


    def add_layer(self, out_dim, activation, kernel_init = 'random_normal',
                 bias_init = 'zeros', first_dim = None, lam = 0):
        '''
        Method for adding layers to feed forward network

        out_dim: number of nodes in the output of the layer.
        activation: activation function class, as defined in activations.py
        kernel_init: initialization scheme for weight matrix/kernel. Can be
                     'random_normal' or 'glorot_uniform'.
        bias_init: init. scheme for bias vector. Can be 'zeros' or 'random_normal'.
        first_dim: dimension of input, must be specified for first layer,
                   otherwise inferred from shape of previous layer.
        lam:  L2 regularization factor, applied to both kernel and bias.
        '''
        if len(self.layers) == 0:
            assert first_dim is not None, 'Must pass first dim. to 1st layer'
        else:
            first_dim = self.layers[-1].kernel.shape[0]

        dim = (out_dim, first_dim)  # dimension of kernel

        if kernel_init == 'random_normal':
            kernel = np.random.normal(0, 0.05, dim)
        elif kernel_init == 'glorot_uniform': # glorot uniform initialization
            bound = np.sqrt(6/(out_dim + first_dim))
            kernel = np.random.uniform(-bound, bound, dim)
        else:
            kernel = np.zeros(dim)

        if bias_init == 'random_normal':
            bias = np.random.normal(0, 0.01, out_dim)
        else:
            bias = np.zeros(out_dim)

        # add layer object to model
        new_layer = Layer(activation, kernel, bias, lam)
        self.layers.append(new_layer)


class Layer(object):
    def __init__(self, activation, kernel_init, bias_init, lam = 0):
        '''
        Layer class. Handles computations done by a layer.

        activation: activation function class as defined in activations.py,
                    applied to output.
        kernel_init: Initial value for kernel matrix.
        bias_init: Initial value for bias vector.
        lam: L2 regularization parameter, applied to both kernel and bias.
        '''
        self.activation = activation
        self.kernel = kernel_init
        self.bias = bias_init
        self.lam = lam

    def __call__(self, x):
        # compute activations
        return self.activation(self.forward(x))

    def forward(self, x):
        # compute activities
        y = np.tensordot(x, self.kernel, axes = [-1, 1]) + self.bias # keep shape of inputs
        return y

    def bias_reg(self):
        # L2 weight regularization of biases, gradient
        return 2*self.bias*self.lam

    def kernel_reg(self):
        # L2 weight regularization of kernel, gradient
        return 2*self.kernel*self.lam
