import matplotlib.pyplot as plt
import numpy as np

from bias_variance import bias_variance_analysis
from k_folds import k_fold_crossvalidation

def grid_search(model_constructor, options, title):
    # method for searching for polynomial degree and shrinkage parameters
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

    # show all heatmaps in same subplot
    fig, axs = plt.subplots(2,2, figsize = (10,10))
    ims = [] # save images for adding colorbar
    map = 'magma'
    ims.append(axs[0,0].imshow(np.log(k_mse), cmap = map, aspect = 'auto'))
    axs[0,0].set_title('10-fold Crossvalidation MSE ($\mathbf{log_{10}}$)', fontweight = 'bold')
    ims.append(axs[0,1].imshow(np.log(b_mse), cmap = map, aspect = 'auto'))
    axs[0,1].set_title('Bootstrap MSE ($\mathbf{log_{10}}$)', fontweight = 'bold')
    ims.append(axs[1,0].imshow(np.log(bias), cmap = map, aspect = 'auto'))
    axs[1,0].set_title('Bias ($\mathbf{log_{10}}$)', fontweight = 'bold')
    ims.append(axs[1,1].imshow(np.log(var), cmap = map, aspect = 'auto'))
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
