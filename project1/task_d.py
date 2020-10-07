import numpy as np
import regression as reg
import grid_search as gs

if __name__ == '__main__':
    lambdas = np.geomspace(1e-5, 100, 8)
    degrees = np.arange(20)
    options = {'n': 1000, 'stddev': 0.0, 'k_folds': 10,
               'n_bootstraps': 100, 'complexity': degrees, 'lambdas': lambdas}
    model = lambda p, lam: reg.RidgeRegression(p, lam)
    gs.grid_search(model, options, './results/task_d_ridge.png')
