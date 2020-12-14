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
    model = models.RBFRNN(options)

    model.compile(loss = loss, optimizer = optimizer, metrics = ['mae'])
    if options.load:
        model.load_weights(name)
    else:
        res = model.fit(x_train, y_train, validation_data = (x_test, y_test),
                        epochs = options.train_steps, batch_size = options.batch_size)
        model.save_weights(name)
    return model

options = params.get_parameters()
name = f'./results/rbfrnn_{options.timesteps}_steps/'

options.out_nodes = 50
options.lr = 1e-5
#options.train_steps = 50
options.output_activation = 'relu'

x_train, x_test, y_train, y_test = ds.load_dataset(f'./datasets/hd_s_{options.timesteps}steps.npz')

options.timesteps = options.timesteps - 1

model = load_model(options)

plot_batch = (x_test[0][:options.batch_size], x_test[1][:options.batch_size])
plot_y = y_test[:options.batch_size]
model(plot_batch, training = False)
pc_states = model.outputs.numpy()
rnn_states = model.rnn_states.numpy()
centers = model.expected_centers.numpy()

visualize.line_activities(pc_states, plot_y, centers, options.out_nodes, name)
visualize.visualize_activities(pc_states[0][None], plot_y[0][None], options.out_nodes, name)
visualize.visualize_activities(rnn_states[0][None], plot_y[0][None], options.out_nodes, name + 'rnn')