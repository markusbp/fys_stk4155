import matplotlib.pyplot as plt
import numpy as np

from bias_variance import bias_variance_analysis
from k_folds import k_fold_crossvalidation

def line_search(model_constructor, options, title):
    '''
        Method for searching only for polynomial degree for OLS (1D grid search)

        model_constructor: function for constructing model, must take
        polynomial degree as argument to run search.
        options: contains various parameters used for training and searching,
        see k_folds.py or bias_variance.py
        title: name of image file to be saved
    '''
    degrees = options['complexity'] # polynomial degrees
    # perform analysis...es
    test_mse, test_bias, test_var = bias_variance_analysis(model_constructor, options)
    k_fold_mse = k_fold_crossvalidation(model_constructor, options)

    degrees = options['complexity']
    plt.semilogy(degrees, k_fold_mse, '-*', label = 'Crossvalidation MSE')
    plt.semilogy(degrees, test_mse,'-o', label = 'Bootstrap MSE')
    plt.semilogy(degrees, test_bias,'-o', label = 'Bias')
    plt.semilogy(degrees, test_var, '-o', label = 'Variance')
    plt.xlabel('Polynomial Degree', fontsize = 12)
    plt.legend(frameon = False)
    plt.savefig(title)
    best_k_fold = np.argmin(k_fold_mse) # return best model parameters
    best_bootstrap = np.argmin(test_mse)
    best_performers = {'Best k_fold mse': k_fold_mse[best_k_fold],
                       'Best k_fold poly': degrees[best_k_fold],
                       'Best bootstrap mse:': test_mse[best_bootstrap],
                       'Best bootstrap poly': degrees[best_bootstrap]}
    return best_performers # best OLS model

def grid_search(model_constructor, options, title):
    '''
        Method for searching for polynomial degree and shrinkage parameters (2D grid search)

        model_constructor: function for constructing model, must take
        polynomial degree and shrinkage parameter as arguments to run search.
        options: contains various parameters used for training and searching,
        see k_folds.py or bias_variance.py
        title: name of image file to be saved
    '''

    degrees = options['complexity'] # polynomial degrees
    lambdas = options['lambdas'] # shrinkage parameters
    n_poly = len(degrees)
    n_lambdas = len(lambdas)

    k_mse = np.zeros((n_lambdas, n_poly)) # k-fold MSE
    b_mse = np.zeros((n_lambdas, n_poly)) # bootstrap MSE
    bias = np.zeros((n_lambdas, n_poly))
    var = np.zeros((n_lambdas, n_poly)) # variance

    for i, lam in enumerate(lambdas):
        model = lambda p: model_constructor(p, lam) # ovveride standard model
        k_fold_mse = k_fold_crossvalidation(model, options)
        bootstrap_mse, bs_bias, bs_var = bias_variance_analysis(model, options)
        k_mse[i] = k_fold_mse # save results for different methods
        b_mse[i] = bootstrap_mse
        bias[i] = bs_bias
        var[i] = bs_var

    # get correct 2D index
    best_k_fold = np.unravel_index(np.argmin(k_mse, axis=None), k_mse.shape)
    best_bootstrap = np.unravel_index(np.argmin(b_mse, axis = None), b_mse.shape)

    best_performers = {'Best k_fold mse': k_mse[best_k_fold],
                       'Best k_fold lam': lambdas[best_k_fold[0]],
                       'Best k_fold poly': degrees[best_k_fold[1]],
                       'Best bootstrap mse:': b_mse[best_bootstrap],
                       'Best bootstrap lam': lambdas[best_bootstrap[0]],
                       'Best bootstrap poly': degrees[best_bootstrap[1]]}

    # show all heatmaps in same subplot
    fig, axs = plt.subplots(2,2, figsize = (12,10))
    ims = [] # save images for adding colorbar
    map = 'magma'
    ims.append(axs[0,0].imshow(np.log10(k_mse), cmap = map, aspect = 'auto'))
    axs[0,0].set_title('10-fold Crossvalidation MSE ($\mathbf{log_{10}}$)', fontweight = 'bold')
    ims.append(axs[0,1].imshow(np.log10(b_mse), cmap = map, aspect = 'auto'))
    axs[0,1].set_title('Bootstrap MSE ($\mathbf{log_{10}}$)', fontweight = 'bold')
    ims.append(axs[1,0].imshow(np.log10(bias), cmap = map, aspect = 'auto'))
    axs[1,0].set_title('Bias ($\mathbf{log_{10}}$)', fontweight = 'bold')
    ims.append(axs[1,1].imshow(np.log10(var), cmap = map, aspect = 'auto'))
    axs[1,1].set_title('Variance ($\mathbf{log_{10}}$)', fontweight = 'bold')
    # set ticks with nicer formatting
    k = 0
    for i in range(2):
        for j in range(2):
            axs[i,j].set_xlabel('Model Complexity (Polynomial Degree)', fontsize = 12)
            axs[i,j].set_xticks(degrees) # polynomial degrees
            axs[i,j].set_xticklabels(degrees)
            axs[i,j].set_yticks(np.arange(n_lambdas)) # lambda values, scientific notation
            axs[i,j].set_yticklabels(['%.1E' % val for val in lambdas])
            axs[i,j].set_ylabel('$\lambda$', fontsize = 12)
            fig.colorbar(ims[k], ax = axs[i,j]) # add colorbar to each pane
            k+= 1

    plt.tight_layout()
    plt.savefig(title)
    return best_performers # return best model parameters
