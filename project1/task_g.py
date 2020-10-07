import numpy as np
import grid_search as gs
import regression as reg

if __name__ == '__main__':
    lambdas = np.geomspace(1e-5, 100, 8)
    degrees = np.arange(5)
    options = {'n': 1000, 'stddev': 0.0, 'k_folds': 10,
               'n_bootstraps': 100, 'complexity': degrees, 'lambdas': lambdas,
               'terrain': True}
               
    model = lambda p: reg.Linear(p)
    gs.grid_search(model, options, './results/task_g_ols.png')

    model = lambda p, lam: reg.RidgeRegression(p, lam)
    gs.grid_search(model, options, './results/task_g_ridge.png')

    model = lambda p, lam: reg.LassoRegression(p, lam)
    gs.grid_search(model, options, './results/task_g_lasso.png')
