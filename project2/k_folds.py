import numpy as np
import matplotlib.pyplot as plt

import tools
import terrain

import regression as reg
import statistics as stats
import franke_function as frank

import sklearn
from sklearn.model_selection import train_test_split


def k_fold_crossvalidation(model_constructor, options):
    '''
    Does k_fold crossvalidation.

    model_constructor: function that returns regression model.
    Must take polynomial degree as argument.
    options: options for analysis, must contain
    n - number of points in franke dataset (if applicable)
    stddev - standard deviation of normal noise to add to Franke function
    complexity - array/list containing polynomials to run through
    k_folds: number of folds, e.g. 5 or 10.
    Can contain optional bool key terrain: this does analysis for terrain data
    '''

    np.random.seed(0)

    n = options['n'] # number of test dataset samples
    stddev = options['stddev'] # noise standard deviation
    complexity = options['complexity'] # polynomial degrees
    k_folds = options['k_folds'] # number of folds

    # get dataset
    try:
        if 'terrain' in options:
            r, labels = terrain.get_dataset()
        else:
            print('Warning: Terrain data not selected, default to test')
            r, labels = frank.get_dataset(n, stddev = stddev)
    except KeyError as e:
        # if terrain not in options, run test dataset
        r, labels = frank.get_dataset(n, stddev = stddev)
    x_train, x_test, y_train, y_test = train_test_split(r, labels, test_size = 0.2)

    # number of samples to go into crossvalidation
    n_train = len(x_train)

    scaler = tools.Scaler(x_train) # scale before crossvalidation
    scaled_x_train = scaler(x_train)
    scaled_x_test = scaler(x_test)

    # shuffle dataset before crossvalidation, using same indices for x and y
    shuffled_inds = np.random.choice(n_train, size = n_train, replace = False)
    shuffled_x = scaled_x_train[shuffled_inds]
    shuffled_y = y_train[shuffled_inds]

    remainder = n_train % k_folds # dividing data into even parts doesn't always work
    # so put the remainder samples in fold test set
    train_size = n_train // k_folds # train fold size
    test_size = n_train // k_folds + remainder # test fold size

    test_loss = np.zeros((len(complexity), test_size, k_folds))

    for i, p in enumerate(complexity):
        model = model_constructor(p) # create model

        for k in range(k_folds):
            # test fold; train on the rest
            test_inds = np.arange(k*train_size, k*train_size + test_size)
            train_mask = np.ones(n_train, dtype = np.bool)
            train_mask[test_inds] = False # train folds are the others

            fold_x_train = shuffled_x[train_mask]
            fold_y_train = shuffled_y[train_mask]
            fold_x_test = shuffled_x[test_inds]
            fold_y_test = shuffled_y[test_inds]

            model.fit(fold_x_train, fold_y_train) # fit on train folds
            test_preds = model.predict(fold_x_test) # predict on test fold
            test_loss[i,:,k] = (fold_y_test - test_preds)**2 # Mean squared error

    mean_test_loss = np.mean(test_loss, axis = (1,2)) # mean over samples, then folds
    return mean_test_loss
