import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import rnn_models as models
import dataset_handler as ds
import model_parameters as params
import visualize


def load_model(options, name):
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

# Train IRNN on cartesian dataset, follows same structure as baseline file
options = params.get_parameters()

x_train, x_test, y_train, y_test = ds.load_dataset(f'./datasets/cartesian{options.timesteps}steps.npz')
options.timesteps = y_test.shape[1]
options.out_nodes = 50
options.lr = 1e-4
options.train_steps = 100
name = f'./results/irnn_{options.timesteps}_steps/'

model = load_model(options, name)

plot_batch = (x_test[0][:options.batch_size], x_test[1][:options.batch_size])
plot_y = y_test[:options.batch_size]
# get states, do all visualization
model(plot_batch, training = False)
states = model.outputs.numpy()
centers = model.expected_centers.numpy()
rnn_states = model.rnn_states.numpy()

visualize.line_activities(states, plot_y, centers, 20, save_loc = name)
visualize.visualize_activities(states[0][None], plot_y[0][None], 20, name)
visualize.visualize_activities(rnn_states[0][None], plot_y[0][None], 20, name, title = 'rnn')

# Evaluate on test set
res = model.evaluate(x_test, y_test, batch_size = options.batch_size)
print(f'Model Test MAE for {options.timesteps} steps : {res[-1]}')


if options.load: # load 10k steps dataset, for high res plotting
    x_train, x_test, y_train, y_test = ds.load_dataset(f'./datasets/cartesian10000steps.npz')
    options.timesteps = y_test.shape[1]
    plot_batch = (x_test[0][:options.batch_size], x_test[1][:options.batch_size])
    plot_y = y_test[:options.batch_size]
    model(plot_batch, training = False)
    pc_states = model.outputs.numpy()
    rnn_states = model.rnn_states.numpy()
    centers = model.expected_centers.numpy()
    # visualize RNN + output activations, in space and output in time
    visualize.visualize_activities(pc_states[0][None], plot_y[0][None], 20, name, title = 'pc_10k')
    visualize.visualize_activities(rnn_states[0][None], plot_y[0][None], 20, name, title= 'rnn_10k')



'''
Run example 1:
Model Test MAE for 99 steps : 0.015829844400286674

Run example 2:
Model Test MAE for 999 steps : 0.040121812373399734

'''
