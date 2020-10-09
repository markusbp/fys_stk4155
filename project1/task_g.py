import numpy as np
import grid_search as gs
import regression as reg

if __name__ == '__main__':
    lambdas = np.geomspace(1e-5, 100, 8)
    degrees = np.arange(20)
    options = {'n': 1000, 'stddev': 0.0, 'k_folds': 10,
               'n_bootstraps': 100, 'complexity': degrees, 'lambdas': lambdas,
               'terrain': True}

    model = lambda p: reg.Linear(p) # search polynomials
    ols_results = gs.line_search(model, options, './results/task_g_ols.png')
    print('OLS Results\n\n', ols_results)

    # model = lambda p, lam: reg.RidgeRegression(p, lam) # search polynomials and lambdas
    # ridge_results = gs.grid_search(model, options, './results/task_g_ridge.png')
    # print('Ridge Results\n\n', ridge_results)
    #
    #
    # model = lambda p, lam: reg.LassoRegression(p, lam) # same, but for lasso
    # lasso_results = gs.grid_search(model, options, './results/task_g_lasso.png')
    # print('Lasso Results\n\n', lasso_results)
