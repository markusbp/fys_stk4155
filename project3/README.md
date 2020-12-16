# Project 3
Contains all files used in project 3:

[rnn_models.py](rnn_models.py): Contains all RNN models used in project,
including IRNN, RBFRNN and a baseline RNN

[model_parameters.py](model_parameters.py): Uses argparse to gather
relevant parameters for models. Used in most modules.

[baseline.py](baseline.py): Trains (optionally loads for plotting)
a baseline RNN for navigation (tf.simpleRNN with default parameters).
Trains on dataset with Cartesian velocity inputs

[irnn.py](irnn.py): Trains (optionally loads for plotting)
an identity initialized RNN (IRNN) for navigation (tf.simpleRNN with
identity recurrent matrix initialization, random normal kernel initialization,
bias zero). Trains on dataset with Cartesian velocity inputs.

[rbfrnn.py](rbfrnn.py): Trains (optionally loads for plotting)
a radial basis function RNN for navigation (custom RNN cell with
speed and head direction "cells"). Trains on dataset with
speed and head direction inputs.

[dataset_handler.py](dataset_handler.py): Creates datasets which consists
of smooth semi-random walks in a square chamber. *Note*: Run to create datasets
for first time use! (creates two datasets, the largest is ~ 250 MB).
Also contains convenience method for loading and preparing datasets.

[visualize.py](visualize.py): Contains methods for plotting average activations
as functions of space, and line plots; i.e. plots of activations as a function of
time; also shows distance.

[plot_device.py](plot_device.py): Contains methods for creating plots to help
explain concepts in article.

[grid_search.py](grid_search.py):
Contains method for searching l2 weight regularization (for rec. weight matrix).

[search_l2.py](search_l2.py):
Module for doing grid search for l2 weight regularization for IRNN.
