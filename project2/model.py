import numpy as np

from tqdm import trange

from gradient_descent import GD

class FFNN(object):
    def __init__(self, loss):
        self.loss = loss
        self.layers = []

    def __call__(self, x):
        for l, layer in enumerate(self.layers):
            x = layer(x) # output is input to next layer
        return x

    def train(self, x, y, lr, mom, epochs = 1, bs = 100):

        steps = len(x)//bs

        n = len(self.layers)
        # set up gradient descent optimizers for each layer, for bias and weigths
        trainer = np.array([[GD(lr, mom), GD(lr, mom)] for i in range(n)])

        for i in trange(epochs, desc = 'Training'):
            for j in range(steps):
                batch_inds = np.random.choice(len(x), bs)
                batch_x = x[batch_inds] # random minibatches
                batch_y = y[batch_inds]
                grad_w, grad_b = self.backprop(batch_x, batch_y)
                print(np.array(grad_w[1]).shape)
                for l, layer in enumerate(self.layers):
                    #print(grad_w[l].shape, l)#.shape, layer.kernel.shape)
                    print(l)
                    print(grad_w[l].shape)
                    print(trainer[l,0].step(layer.kernel, grad_w[l], i))
                    layer.kernel = trainer[l,0].step(layer.kernel, grad_w[l], i)
                    layer.bias = trainer[l,1].step(layer.bias, grad_b[l], i)

    def backprop(self, x, y):
        n = len(self.layers)
        a = [0 for i in range(n+1)] # activations
        grad = [0 for i in range(n)] # derivative of activation function
        g = [0 for i in range(n)] # gamma, "error"

        grad_w = [0 for i in range(n)]
        grad_b = [0 for i in range(n)]

        a[0] = x
        # forward pass
        for l, layer in enumerate(self.layers):
            z = layer.forward(x)
            grad[l] = layer.activation.gradient(z)
            x = layer.activation(z)
            a[l+1] = x

        # backward pass
        g[-1] = self.loss.gradient(a[-1], y)*grad[-1] # output "error"
        for l in range(1, len(self.layers) + 1):
            if l == 1:
                pass
            else:
                g[-l] = np.tensordot(g[-l+1], self.layers[-l+1].kernel.T, axes = [-1, 1])*grad[-l]
                grad_w[-l] = g[-l][:,:, None]*a[-l-1][:,None,:]
                grad_b[-l] = g[-l]
            print(-l)

        return grad_w, grad_b

    def add_layer(self, out_dim, activation, kernel_init = 'random_normal',
                 bias_init = 'zeros', first_dim = None):

        if len(self.layers) == 0:
            assert first_dim is not None, 'Must pass first dim. to 1st layer'
        else:
            first_dim = self.layers[-1].kernel.shape[0]

        dim = (out_dim, first_dim) # [x, units]-> [x]

        if kernel_init == 'random_normal':
            kernel = np.random.normal(0, 0.01, dim)
        else:
            kernel = np.zeros(dim)

        if bias_init == 'random':
            pass
        else:
            bias = np.zeros(out_dim)

        new_layer = Layer(dim, activation, kernel, bias)
        self.layers.append(new_layer)

class Layer(object):
    def __init__(self, dim, activation, kernel_init, bias_init):
        self.dim = dim
        self.activation = activation
        self.kernel = kernel_init
        self.bias = bias_init

    def __call__(self, x):
        return self.activation(self.forward(x))

    def forward(self, x):
        y = np.tensordot(x, self.kernel, axes = [-1, 1]) + self.bias # keep shape of inputs
        return y
