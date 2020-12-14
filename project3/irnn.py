import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import rnn_models as models
import dataset_handler as ds
import model_parameters as params
import visualize


def load_model(options):
    loss = tf.keras.losses.MSE
    optimizer = tf.keras.optimizers.Adam(options.lr)
    model = models.EgoRNN(options)

    model.compile(loss = loss, optimizer = optimizer, metrics = ['mae'])
    if options.load:
        model.load_weights(name)
    else:
        res = model.fit(x_train, y_train, validation_data = (x_test, y_test),
                        epochs = options.train_steps, batch_size = options.batch_size)
        model.save_weights(name)
    return model
# Train baseline RNN, tanh activation

x_train, x_test, y_train, y_test = ds.load_dataset('./datasets/cartesian1000steps.npz')

options = params.get_parameters()
options.timesteps = y_test.shape[1]
options.out_nodes = 50

options.lr = 1e-5
options.train_steps = 50

model = load_model(options)

plot_batch = (x_test[0][:options.batch_size], x_test[1][:options.batch_size])
plot_y = y_test[:options.batch_size]
model(plot_batch, training = False)
states = model.outputs.numpy()
centers = model.expected_centers.numpy()

name = f'./results/irnn_{options.timesteps}_steps/'
model.save_weights(name)
visualize.line_activities(states, plot_y, centers, options.out_nodes, save_loc = name)
visualize.visualize_activities(states[0][None], plot_y[0][None], options.out_nodes, name)
