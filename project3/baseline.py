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
        model.save_weights(name) # save result
    return model

# Train baseline RNN, tanh activation
# get parameters for model
options = params.get_parameters()

# load dataset
x_train, x_test, y_train, y_test = ds.load_dataset(f'./datasets/cartesian{options.timesteps}steps.npz')

options.timesteps = y_test.shape[1] # actual number of timesteps is one less (starting step)
options.out_nodes = 50 # "place cells"
options.train_steps = 100 # number of training epochs
options.lr = 5e-5 # learning rate

name = f'./results/baseline_{options.timesteps}_steps/'

model = load_model(options, name)  # load model

# plot selection of place cell activities in time
plot_batch = (x_test[0][:options.batch_size], x_test[1][:options.batch_size])
plot_y = y_test[:options.batch_size]
model(plot_batch) # run to get PC states
states = model.outputs.numpy() # place cell activations ?
rnn_states = model.rnn_states.numpy() # grid cell activations ?
centers = model.expected_centers.numpy() # place cell centers

# plot all activations
visualize.line_activities(states, plot_y, centers, options.out_nodes, name)
visualize.visualize_activities(states[0][None], plot_y[0][None], options.out_nodes, name)
visualize.visualize_activities(rnn_states[0][None], plot_y[0][None], options.out_nodes, name, title = 'rnn')

# Evaluate on test set
res = model.evaluate(x_test, y_test, batch_size = options.batch_size)
print(f'Model Test MAE for {options.timesteps} steps : {res[-1]}')

'''
Run Example 1:
Model Test MAE for 99 steps : 0.015122444368898869

Run Example 2:
Model Test MAE for 999 steps : 0.25813546776771545
'''
