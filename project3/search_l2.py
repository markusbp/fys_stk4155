import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import rnn_models as models
import dataset_handler as ds
import model_parameters as params
import visualize
import grid_search

# search l2 weight reg for IRNN
options = params.get_parameters()

options.out_nodes = 50
options.train_steps = 50

lr = 1e-4

irnn = lambda opt: models.EgoRNN(opt) # best performer
destination = './results/l2_search/' # save here
dataset = './datasets/cartesian1000steps.npz'
grid_search.search_l2(destination, dataset, irnn, options)
