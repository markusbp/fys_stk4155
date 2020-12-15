import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import rnn_models as models
import dataset_handler as ds
import model_parameters as params
import visualize
import grid_search

options = params.get_parameters()
options.timesteps = y_test.shape[1]
options.out_nodes = 30

lr = 1e-5
epochs = 100

baseline = lambda opt: models.BaseLineRNN(opt)
irnn = lambda opt: models.EgoRNN(opt)
rbfrnn = lambda opt: models.RBFRNN(opt)
destination = './results/dropout_search/'
dataset = './datasets/cartesian100steps.npz'
grid_search.search_dropout(destination, dataset, model, options)
