import os
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

import model_parameters as params
import dataset_handler as ds
import visualize

def search_dropout(path, dataset, model_constructor, options):

    if not os.path.isdir(path):
        os.makedirs(path)

    x_train, x_test, y_train, y_test, r_train, r_test = ds.load_dataset(dataset)

    n_plot = 5000
    plot_x = (x_test[0][:n_plot], x_test[1][:n_plot])
    plot_r = r_test[:n_plot]

    epochs = options.train_steps

    loss = tf.keras.losses.MSE
    optimizer = tf.keras.optimizers.Adam(options.lr)

    drops = np.array([0, 0.1, 0.25, 0.5])
    mae = np.zeros(len(drops))

    for i, rate in enumerate(drops):

        options.dropout_rate = rate
        model = model_constructor(options)

        model.compile(loss = loss, optimizer = optimizer, metrics = ['mae'])

        history = model.fit(x_train, y_train, epochs = epochs)

        results = model.evaluate(x_test, y_test)

        mae[i] = results[-1] # mean absolute error

        model(plot_x, training = False) # run to get states as np arrays
        states = model.outputs.numpy()

        name = f'{path}_rate_{rate}.png'
        visualize.visualize_activities(states, plot_r, name)

    plt.plot(drops, mae, 'o', linewidth = 0.75)
    plt.xlabel('Dropout Rate', fontsize = 12)
    plt.ylabel('MAE', fontsize = 12)
    plt.savefig(f'{path}mae_drop.png')
    plt.close()

def search_beta(path, dataset, model_constructor, options):

    if not os.path.isdir(path):
        os.makedirs(path)

    x_train, x_test, y_train, y_test, r_train, r_test = ds.load_dataset(dataset)

    n_plot = 5000
    plot_x = (x_test[0][:n_plot], x_test[1][:n_plot])
    plot_r = r_test[:n_plot]

    epochs = options.train_steps

    loss = tf.keras.losses.MSE
    optimizer = tf.keras.optimizers.Adam(options.lr)

    betas = np.array([1, 10, 100, 1000])
    mae = np.zeros(len(betas))

    for i, beta in enumerate(betas):

        options.beta = beta
        model = model_constructor(options)

        model.compile(loss = loss, optimizer = optimizer, metrics = ['mae'])

        history = model.fit(x_train, y_train, epochs = epochs)
        results = model.evaluate(x_test, y_test)

        mae[i] = results[-1] # mean absolute error

        model(plot_x, training = False) # run to get states as np arrays
        states = model.outputs.numpy()

        name = f'{path}_beta_{beta}.png'
        visualize.visualize_activities(states, plot_r, name)

    plt.plot(betas, mae, 'o', linewidth = 0.75)
    plt.xlabel('Beta', fontsize = 12)
    plt.ylabel('MAE', fontsize = 12)
    plt.savefig(f'{path}mae_beta.png')
    plt.close()
