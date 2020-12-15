import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import rnn_models as models
import dataset_handler as ds
import model_parameters as params
import visualize


def load_model(options, name):
    loss = tf.keras.losses.MSE # xy coordinates, MSE makes sense
    optimizer = tf.keras.optimizers.Adam(options.lr)
    model = models.BaseLineRNN(options)

    # keep track of MAE; easy to read off navigation error!
    model.compile(loss = loss, optimizer = optimizer, metrics = ['mae'])

    if options.load:
        model.load_weights(name)
        # train model
    else:
        res = model.fit(x_train, y_train, validation_data = (x_test, y_test),
                        epochs = options.train_steps, batch_size = options.batch_size)
        model.save_weights(name)
    return model

# Train baseline RNN, tanh activation
# get parameters for model
options = params.get_parameters()

# load dataset
x_train, x_test, y_train, y_test = ds.load_dataset(f'./datasets/cartesian{options.timesteps}steps.npz')

options.timesteps = y_test.shape[1]
options.out_nodes = 50
lr = 1e-5 # learning rate
name = './results/baseline/'

model = load_model(options, name)

# plot selection of place cell activities in time
plot_batch = (x_test[0][:options.batch_size], x_test[1][:options.batch_size])
plot_y = y_test[:options.batch_size]
model(plot_batch) # run to get PC states
states = model.outputs.numpy()
centers = model.expected_centers.numpy()
visualize.line_activities(states, plot_y, centers, options.out_nodes, name)
visualize.visualize_activities(states[0][None], plot_y[0][None], options.out_nodes, name)
