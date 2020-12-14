import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import rnn_models as models
import dataset_handler as ds
import model_parameters as params
import visualize
import grid_search
# Train baseline RNN, tanh activation


options = params.get_parameters()
options.timesteps = y_test.shape[1]
options.out_nodes = 30
options.output_activation = 'relu'

lr = 1e-5
epochs = 100

model = lambda opt: models.EgoRNN(opt)

destination = './results/dropout_search/'
dataset = './datasets/cartesian100steps.npz'
grid_search.search_dropout(destination, dataset, model, options)
