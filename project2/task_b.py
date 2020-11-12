import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import franke_function as frank
import regression as reg
from gradient_descent import GD

from model import FFNN
import activations
import loss_functions

import seaborn as sns
import matplotlib.pyplot as plt

def bootstrap_mse(model, x_test, y_test, n_bootstraps = 100):
    err = 0
    for i in range(n_bootstraps):
        b_x, b_y = resample(x_test, y_test)
        err += np.mean((model(b_x) - b_y)**2)
    return err/n_bootstraps

def bootstrap_r2(model, x_test, y_test, n_bootstraps = 100):
    r2 = 0
    for i in range(n_bootstraps):
        b_x, b_y = resample(x_test, y_test)
        y_bar = np.mean(b_y)
        pred = model(b_x)
        r2 += 1 - np.sum( (pred - b_y)**2 )/np.sum((b_y - y_bar)**2)
    return r2/n_bootstraps

def verify(r_train, y_train, r_test, y_test):

    mse = loss_functions.MSE()
    relu = activations.Relu()
    linear = activations.Linear()
    model = FFNN(mse)
    model.add_layer(128, relu, first_dim = r_train.shape[-1])
    model.add_layer(1, linear)

    batch_size = 10
    epochs = 1000
    lr = 1e-3

    train_err, test_err = model.train(r_train, y_train[:, None], lr = lr,
                                      epochs = epochs, mom = 0, bs = batch_size,
                                      val_data = (r_test, y_test[:,None]), decay = False)

    input_layer = tf.keras.layers.Dense(128, activation = 'relu', kernel_initializer = 'random_normal', bias_initializer = 'zeros')
    output_layer = tf.keras.layers.Dense(1, activation = None, kernel_initializer = 'random_normal', bias_initializer = 'zeros')
    model = tf.keras.Sequential([input_layer, output_layer])

    my_optimizer = tf.keras.optimizers.SGD(learning_rate = lr, decay = 0)

    model.compile(loss = 'mse', optimizer = my_optimizer)
    history = model.fit(r_train, y_train[:,None], validation_data = (r_test, y_test[:,None]),
               batch_size = batch_size, epochs = epochs)

    plt.figure(figsize = (6.5, 5))
    plt.semilogy(train_err, 'b-', label = 'Train', linewidth = 0.75)
    plt.semilogy(test_err, 'r-', label = 'Test', linewidth = 0.75)
    plt.semilogy(history.history['val_loss'], label = 'Test, tf', linewidth = 0.75)
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('MSE', fontsize = 12)
    plt.legend(frameon = False)
    plt.tight_layout()
    plt.savefig('./results/task_b.png')

def search(r_train, y_train, r_test, y_test, epochs = 100):

    learning_rates = np.geomspace(1e-4, 0.1, 4)
    lams = np.geomspace(1e-4, 0.1, 4)

    test_err = np.zeros((len(lams), len(learning_rates)))
    test_r2 = np.zeros((len(lams), len(learning_rates)))

    mse = loss_functions.MSE()
    relu = activations.Relu()
    linear = activations.Linear()

    batch_size = 10

    for i, lam in enumerate(lams):
        for j, lr in enumerate(learning_rates):

            model = FFNN(mse)
            model.add_layer(128, relu, first_dim = r_train.shape[-1], lam = lam)
            model.add_layer(1, linear, lam = lam)

            train_e, test_e = model.train(r_train, y_train[:, None], lr = lr,
                                              epochs = epochs, mom = 0, bs = batch_size,
                                              val_data = (r_test, y_test[:,None]), decay = False)

            test_err[i, j] = bootstrap_mse(model, r_test, y_test[:,None])
            test_r2[i, j] = bootstrap_r2(model, r_test, y_test[:,None])

    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    sns.heatmap(test_err, annot=True, ax=ax[0], cmap="magma",
                xticklabels = learning_rates, yticklabels = lams)
    ax[0].set_title("Test MSE")
    ax[0].set_xlabel("Learning Rate", fontsize = 12)
    ax[0].set_ylabel("$\lambda$", fontsize = 12)

    sns.heatmap(test_r2, annot=True, ax=ax[1], cmap="magma",
                xticklabels = learning_rates, yticklabels = lams)

    ax[1].set_title("Test R2 Score")
    ax[1].set_xlabel("Learning Rate", fontsize = 12)
    ax[1].set_ylabel("$\lambda$", fontsize = 12)
    plt.savefig('./results/task_b_search.png')

if __name__ == '__main__':

    samples = 1000
    data, labels = frank.get_dataset(samples)
    r_train, r_test, y_train, y_test = train_test_split(data, labels)

    mu = np.mean(r_train)
    r_train = r_train - mu
    r_test = r_test - mu

    verify(r_train, y_train, r_test, y_test)
    search(r_train, y_train, r_test, y_test)
