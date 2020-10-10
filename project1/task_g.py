import numpy as np
import grid_search as gs
import regression as reg

if __name__ == '__main__':
    lambdas = np.geomspace(1e-5, 100, 8)
    degrees = np.arange(20)
    options = {'n': 1000, 'stddev': 0.0, 'k_folds': 10,
               'n_bootstraps': 100, 'complexity': degrees, 'lambdas': lambdas,
               'terrain': True}

    # model = lambda p: reg.Linear(p) # search polynomials
    # ols_results = gs.line_search(model, options, './results/task_g_ols.png')
    # print('OLS Results:\n\n', ols_results)
    #
    # model = lambda p, lam: reg.RidgeRegression(p, lam) # search polynomials and lambdas
    # ridge_results = gs.grid_search(model, options, './results/task_g_ridge.png')
    # print('Ridge Results:\n\n', ridge_results)

    options['degrees'] = np.arange(10, 30)
    options['lambdas'] = np.geomspace(1e-6, 1e-1, 6)
    model = lambda p, lam: reg.LassoRegression(p, lam) # same, but for lasso
    lasso_results = gs.grid_search(model, options, './results/task_g_lasso.png')
    print('Lasso Results:\n\n', lasso_results)

    # example run

    '''
    OLS Results:

    'Best k_fold mse': 0.007977875063271965,
    'Best k_fold poly': 14, 'Best bootstrap mse:': 0.010117379757707415,
    'Best bootstrap poly': 8

    Ridge Results:

    'Best k_fold mse': 0.007396532235082904, 'Best k_fold lam': 0.0001,
    'Best k_fold poly': 14, 'Best bootstrap mse:': 0.008153912222927344,
    'Best bootstrap lam': 0.1, 'Best bootstrap poly': 11

    # Note: Lasso did not converge for many values of lambda
    Lasso Results:
    'Best k_fold mse': 0.010149498359619303, 'Best k_fold lam': 1e-05,
    'Best k_fold poly': 15, 'Best bootstrap mse:': 0.009114644126935119,
    'Best bootstrap lam': 1e-05, 'Best bootstrap poly': 18

    '''
