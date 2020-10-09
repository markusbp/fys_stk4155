import numpy as np
import regression as reg
import grid_search as gs

if __name__ == '__main__':
    lambdas = np.geomspace(1e-5, 100, 8)
    degrees = np.arange(20)
    options = {'n': 1000, 'stddev': 0.0, 'k_folds': 10,
               'n_bootstraps': 100, 'complexity': degrees, 'lambdas': lambdas}
    model = lambda p, lam: reg.LassoRegression(p, lam)
    results = gs.grid_search(model, options, './results/task_e_lasso.png')
    print('Results:\n\n', results)

    # example run, note: lasso did not converge for many values of lambda
    '''
        'Best k_fold mse': 0.0004475753878630241, 'Best k_fold lam': 1e-05,
        'Best k_fold poly': 15, 'Best bootstrap mse:': 0.0005032690455742847,
        'Best bootstrap lam': 1e-05, 'Best bootstrap poly': 19}
    '''
