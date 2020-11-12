import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from model import FFNN
import activations
import loss_functions

import matplotlib.pyplot as plt

# Load data from https://www.openml.org/d/554
x, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home = './datasets')

train_samples = 60000
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_samples, test_size=10000)

# normalize data:
x_train = x_train/255.0
x_test = x_test/255.0
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

lr = 1e-4

# convert integer labels to one_hot
y = np.zeros((len(x_train), 10))
y_t = np.zeros((len(x_test), 10))
for i in range(len(x_train)):
    y[i, y_train[i]] = 1

for i in range(len(x_test)):
    y_t[i, y_test[i]] = 1

epochs = 50 # antall ganger vi kjører gjennom hele datasettet
batch_size = 50

loss = loss_functions.SoftmaxCrossEntropy()
softmax = activations.Softmax()

lams = [1e-4, 1e-3, 1e-2, 0.1]

test_losses = np.zeros((len(lams), epochs))

for i, lam in enumerate(lams):

    model = FFNN(loss)
    model.add_layer(10, softmax, first_dim = x_train.shape[-1],
                    kernel_init = 'glorot_uniform', bias_init = 'zeros', lam = lam)

    train_result, test_result = model.train(x_train, y, lr, mom = 0,
                                            epochs = epochs, bs = batch_size,
                                            val_data = (x_test, y_t), metrics = ['loss', 'accuracy'] )

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
# loss , accuracy
for i, name in enumerate(['Crossentropy', 'Accuracy']):
    ax[i].plot(train_result[:,i], label =  'Train', linewidth = 0.75)
    ax[i].plot(test_result[:,i], label = 'Test', linewidth = 0.75)
    ax[i].legend(frameon = False)
    ax[i].set_xlabel('Epoch', fontsize = 12)
    ax[i].set_title(name, fontsize = 14)
    ax[i].legend(frameon = False)

plt.savefig('./results/task_e.png')
