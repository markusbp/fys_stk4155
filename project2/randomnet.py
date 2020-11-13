import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import regression as reg
from gradient_descent import GD
from model import FFNN
import activations
import loss_functions
import franke_function as frank

# build a random neural network for regression of Franke function dataset
n = 1000
data, labels = frank.get_dataset(n)
r_train, r_test, y_train, y_test = train_test_split(data, labels)
# scale dataset
mean = np.mean(r_train)
r_train = r_train - mean
r_test = r_test - mean

mse = loss_functions.MSE() # loss

# could have any activation function
sigmoid = activations.Sigmoid()
relu = activations.Relu()
leaky_relu = activations.LeakyRelu()
linear = activations.Linear()

hidden_activations = [sigmoid, relu, leaky_relu, linear]
names = ['Sigmoid', 'Relu', 'Leaky Relu', 'Linear']

# and any initialization
kernel_inits = ['glorot_uniform', 'random_normal']
bias_inits = ['random_normal', 'zeros']

# training parameters
batch_size = 10
n_models = 10

max_layers = 10 # maximum number of layers

# initial stats
best_error = 1
best_model = None
best_params = []

for i in range(n_models):

    model = FFNN(mse) # new model

    n_layers = np.random.randint(1, max_layers) # random number of layers
    params = [] # save model parameters

    for j in range(n_layers):
        if j == 0:
            first_dim = r_train.shape[-1] # set input shape
        else:
            first_dim = None

        # randomly draw number of nodes, activation, and initializations
        n_nodes = np.random.choice(100) # max 100 nodes in a layer
        ind = np.random.choice(len(hidden_activations))
        act_name = names[ind]
        act = hidden_activations[ind]
        bias_init = np.random.choice(bias_inits)
        kernel_init = np.random.choice(kernel_inits)
        model.add_layer(n_nodes, act, first_dim = first_dim,
                        kernel_init = kernel_init, bias_init = bias_init)
        # save params
        params.append([n_nodes, act_name, [bias_init, kernel_init]])
    # linear output layer
    model.add_layer(1, linear, kernel_init = 'glorot_uniform', bias_init = 'zeros')

    train_err, test_err = model.train(r_train, y_train[:, None], lr = 1e-3,
                                      epochs = 500, mom = 0, bs = batch_size,
                                      val_data = (r_test, y_test[:,None]))

    if test_err[-1,0] < best_error: # save best performer
        best_error = test_err[-1,0]
        test_error = test_err
        train_error = train_err

        best_model = model
        best_params = params

print('Best model parameters:', best_params, '. With', len(best_params), 'hidden layers.')
print('Best test error:', best_error)

plt.plot(test_error, label = 'Test', linewidth = 0.75)
plt.plot(train_error, label = 'Train', linewidth = 0.75)
plt.xlabel('Epoch', fontsize = 12)
plt.ylabel('MSE', fontsize = 12)
plt.title('RandomNet', fontweight = 'bold')
plt.legend(frameon = False)
plt.tight_layout()
plt.savefig('./results/random.png')


# Example run:
# Best model parameters: [[35, 'Leaky Relu', ['zeros', 'glorot_uniform']],
# [88, 'Relu', ['random_normal', 'glorot_uniform']], [45, 'Linear',
# ['random_normal', 'random_normal']], [21, 'Leaky Relu', ['zeros', 'glorot_uniform']],
# [50, 'Leaky Relu', ['random_normal', 'glorot_uniform']], [67, 'Relu', ['zeros', 'glorot_uniform']],
# [22, 'Leaky Relu', ['random_normal', 'glorot_uniform']]] . With 7 hidden layers.
# Best test error: 0.057070955211923305
