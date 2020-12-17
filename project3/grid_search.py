import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import model_parameters as params
import dataset_handler as ds
import visualize

def search_l2(path, dataset, model_constructor, options):
    # search l2 weight reg.
    if not os.path.isdir(path):
        os.makedirs(path)
    # load dataset
    x_train, x_test, y_train, y_test = ds.load_dataset(dataset)
    # also plot activations for all trained models

    plot_x = (x_test[0][:options.batch_size], x_test[1][:options.batch_size])
    plot_y = y_test[:options.batch_size] # plot batch

    epochs = options.train_steps

    loss = tf.keras.losses.MSE
    optimizer = tf.keras.optimizers.Adam(options.lr)

    l2s = np.geomspace(1e-2, 1, 3) # l2 values to search
    mae = np.zeros(len(l2s)) # save mean absolute error

    for i, l2 in enumerate(l2s):

        options.l2 = l2
        model = model_constructor(options)

        model.compile(loss = loss, optimizer = optimizer, metrics = ['mae'])
        # train and evaluate
        history = model.fit(x_train, y_train, epochs = epochs, batch_size = options.batch_size)
        results = model.evaluate(x_test, y_test, batch_size = options.batch_size)

        mae[i] = results[-1] # mean absolute error

        model(plot_x, training = False) # run to get states as np arrays
        states = model.outputs.numpy()
        rnn_states = model.rnn_states.numpy()
        name = f'l2_{l2}'

        visualize.visualize_activities(states[0][None], plot_y[0][None], options.out_nodes, path, title = name + '_pc')
        visualize.visualize_activities(rnn_states[0][None], plot_y[0][None], options.out_nodes, path, title = name + '_rnn')

    # plot and save errors
    plt.semilogx(l2s, mae, 'r-o', linewidth = 0.75)
    plt.xlabel('L2', fontsize = 12)
    plt.ylabel('MAE', fontsize = 12)
    plt.savefig(f'{path}mae_l2.png')
    plt.close()
