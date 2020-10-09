import numpy as np
import matplotlib.pyplot as plt

import tools
import terrain
import regression as reg
import statistics as stats
import franke_function as frank

import sklearn
from sklearn.model_selection import train_test_split

def bias_variance_analysis(model_constructor, options):
    # Perform bias-variance analysis
    np.random.seed(0)
    # get analysis options
    n = options['n'] # number of samples for test dataset
    stddev = options['stddev'] # noise
    complexity = options['complexity'] # maximum degrees of polynomials to be used
    n_bootstraps = options['n_bootstraps']

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

    input_scaler = tools.Scaler(x_train) # Do scaling before bootstrapping
    scaled_x_train = input_scaler(x_train)
    scaled_x_test = input_scaler(x_test)

    test_mse = np.zeros(len(complexity)) # init. statistics
    test_var = np.zeros(len(complexity))
    test_bias = np.zeros(len(complexity))
    
    # almost all tasks involve running over polynomials, so why not do it here
    for i, p in enumerate(complexity): # grid search polynomial degree
        test_preds = np.zeros((len(y_test), n_bootstraps))
        model = model_constructor(p) # create model

        for b in range(n_bootstraps): # bootstrap resample
            boot_x_train, boot_y_train = sklearn.utils.resample(scaled_x_train, y_train)
            model.fit(boot_x_train, boot_y_train) # train model
            test_preds[:,b] = model.predict(scaled_x_test)

        # compute all quantities
        average_prediction = np.mean(test_preds, axis = -1)
        test_mse[i] = np.mean(np.mean((y_test[:,None] - test_preds)**2, axis=1, keepdims=True))
        test_bias[i] = np.mean((y_test- average_prediction)**2)
        test_var[i] = np.mean( np.var(test_preds, axis=1, keepdims=True))

    return test_mse, test_bias, test_var
