import numpy as np
import matplotlib.pyplot as plt

import tools
import regression as reg
import statistics as stats
import franke_function as frank

import sklearn
from sklearn.model_selection import train_test_split

from bias_variance import bias_variance_analysis

def task_b_1():
    n = 1000 # 1000 test data points
    complexity = 20

    r, labels = frank.get_dataset(n, stddev = 1) # Noisy data
    x_train, y_train, x_test, y_test = tools.split_data(r, labels)

    input_scaler = tools.Scaler(x_train)
    scaled_train_x = input_scaler(x_train)
    scaled_test_x = input_scaler(x_test)

    train_mse = np.zeros(complexity)
    test_mse = np.zeros(complexity)

    # Recreate 2.11 in Hastie et al. Compute test/train MSE up to 20th order polynomial
    for p in range(complexity):
        model = reg.Linear(p) # linear regression

        model.fit(scaled_train_x, y_train)
        # compute train/test MSE
        train_preds = model.predict(scaled_train_x)
        test_preds = model.predict(scaled_test_x)
        # Compute statistics
        train_mse[p] = stats.mse(train_preds, y_train)
        test_mse[p] = stats.mse(test_preds, y_test)

    plt.semilogy(np.arange(complexity), test_mse, '-o', label = 'Test', linewidth = 1)
    plt.semilogy(np.arange(complexity), train_mse, '-o', label = 'Train', linewidth = 1)
    plt.xticks(np.arange(0,complexity, 2))
    plt.legend(frameon = False)

    plt.xlabel('Model Complexity (Polynomial degree)', fontsize = 12)
    plt.ylabel('Mean Squared Error', fontsize = 12)
    plt.savefig('./results/task_b_train_test_mse.png')
    plt.close()

def task_b_2():
    options = {'n': 1000, 'complexity': np.arange(20), 'stddev': 0, 'n_bootstraps': 100}
    model = lambda p: reg.Linear(p) # model constructor function
    test_mse, test_bias, test_var = bias_variance_analysis(model, options)
    # run bias variance analysis
    title = './results/task_b_bias_variance.png'

    degrees = options['complexity']
    plt.semilogy(degrees, test_mse,'-o', label = 'MSE', linewidth = 1)
    plt.semilogy(degrees, test_bias,'-o', label = 'Bias', linewidth = 1)
    plt.semilogy(degrees, test_var, '-o', label = 'Variance', linewidth = 1)
    plt.xlabel('Polynomial Degree', fontsize = 12)
    plt.legend(frameon = False)
    plt.savefig(title)

if __name__ == '__main__':
    task_b_1()
    task_b_2()
