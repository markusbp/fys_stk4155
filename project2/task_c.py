import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import regression as reg
from gradient_descent import GD
from model import FFNN
import activations
import loss_functions
import franke_function as frank


n = 1000
data, labels = frank.get_dataset(n)
r_train, r_test, y_train, y_test = train_test_split(data, labels)

mean = np.mean(r_train)
r_train = r_train - mean
r_test = r_test - mean

mse = loss_functions.MSE()

sigmoid = activations.Sigmoid()
relu = activations.Relu()
leaky_relu = activations.LeakyRelu()
linear = activations.Linear()
funcs = [sigmoid, relu, leaky_relu, linear]

fig, axs = plt.subplots(2,2, figsize = (10, 10))
row = [0, 0, 1, 1]
col = [0, 1, 0, 1]
names = ['Sigmoid', 'Relu', 'Leaky Relu', 'Linear']

batch_size = 10
best_test = 1

for activation, name, row, col in zip(funcs, names, row, col):
    for init in ['random_normal', 'glorot_uniform']:

        model = FFNN(mse)
        model.add_layer(128, activation, first_dim = r_train.shape[-1], kernel_init = init, bias_init = init)
        model.add_layer(1, linear, kernel_init = init, bias_init = init)

        train_err, test_err = model.train(r_train, y_train[:, None], lr = 1e-3,
                                          epochs = 500, mom = 0, bs = batch_size,
                                          val_data = (r_test, y_test[:,None]))
        if np.amin(test_err) < best_test:
            best_test = np.amin(test_err)
            best_model = name + ' ' + init

        axs[row, col].plot(train_err, label = 'Train, ' + init.replace('_', ' ') + ' init.', linewidth = 0.75)
        axs[row, col].plot(test_err, label = 'Test, ' + init.replace('_', ' ') + ' init.' , linewidth = 0.75)
        axs[row, col].legend(frameon = False)
    axs[row, col].set_xlabel('Epoch', fontsize = 12)
    axs[row, col].set_ylabel('MSE', fontsize = 12)
    axs[row, col].set_title(name, fontweight = 'bold')

plt.tight_layout()
plt.savefig('./results/task_c.png')
print('Best model test MSE', best_test, 'Best model:', best_model)
