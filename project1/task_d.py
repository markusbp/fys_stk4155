import numpy as np
import matplotlib.pyplot as plt

import regression as reg
import grid_search as gs

import tools
import regression as reg
import statistics as stats
import franke_function as frank

from sklearn.model_selection import train_test_split


def shrinkage_analysis():

    p = 3 # Degree of (2D) polynomial to use

    n = 1000 # number of data points
    sig = 1.96 # significance leve, corresponds to 95% CI
    fig, axs = plt.subplots(1, 2, figsize = (10,5))

    for i, std in enumerate([0, 1]):

        r, labels = frank.get_dataset(n, stddev = std)

        x_train, x_test, y_train, y_test = train_test_split(r, labels, test_size = 0.2)

        input_scaler = tools.Scaler(x_train)
        scaled_train_x = input_scaler(x_train)
        scaled_test_x = input_scaler(x_test)

        betas = []
        cis = []
        lambdas = np.geomspace(1e-4, 1000, 11)
        for lam in lambdas:

            model = reg.RidgeRegression(p, lam)

            model.fit(scaled_train_x, y_train)

            train_preds = model.predict(scaled_train_x)
            test_preds = model.predict(scaled_test_x)
            # Compute statistics
            train_mse = stats.mse(train_preds, y_train)
            test_mse = stats.mse(test_preds, y_test)
            r2 = stats.r2_score(test_preds, y_test)

            # find variance of Ridge estimator
            # estimate error as 1/(N-p-1)*sum of residuals
            sigma2 = 1/(n-p-1)*test_mse*len(y_test) # Test MSE = 1/n_test*sum of residuals
            X = model.design_matrix(scaled_test_x) # get design matrix

            ridge = X.T@X + lam*np.eye(X.shape[-1])
            var_beta_ridge = sigma2*np.linalg.inv(ridge)@X.T@X@np.linalg.inv(ridge).T

            var_coeffs = np.diag(var_beta_ridge)

            errorbars = sig*np.sqrt(var_coeffs)
            cis.append(errorbars)
            betas.append(model.beta_)

        # plot coefficients with errors vs. lambda
        betas = np.array(betas).T
        cis = np.array(cis).T # confidence interval values
        axs[i].set_title(f'$S_D = {std}$', fontweight = 'bold')
        for j in range(len(betas)):
            axs[i].errorbar(lambdas, betas[j], yerr = cis[j],
                         fmt = '-o')
            axs[i].set_xscale('log')
            axs[i].set_xlabel('$\lambda$', fontsize = 12)
            axs[i].set_ylabel('Coefficient Value', fontsize = 12)
    plt.tight_layout()
    plt.savefig('./results/task_d_shrinkage.png')

def ridge_search():
    lambdas = np.geomspace(1e-5, 100, 8)
    degrees = np.arange(20)
    options = {'n': 1000, 'stddev': 0.0, 'k_folds': 10,
               'n_bootstraps': 100, 'complexity': degrees, 'lambdas': lambdas}
    model = lambda p, lam: reg.RidgeRegression(p, lam)
    results = gs.grid_search(model, options, './results/task_d_ridge.png')
    print('Results:\n\n', results)

if __name__ == '__main__':
    shrinkage_analysis()
    ridge_search()

    # example run output
    '''
    'Best k_fold mse': 9.890978384681159e-06, 'Best k_fold lam': 0.0001,
    'Best k_fold poly': 18, 'Best bootstrap mse:': 3.2392411699815e-05,
    'Best bootstrap lam': 0.0001, 'Best bootstrap poly': 13
    '''
