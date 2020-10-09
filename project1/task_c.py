import numpy as np
import matplotlib.pyplot as plt

import regression as reg
from k_folds import k_fold_crossvalidation

if __name__ == '__main__':
    model = lambda p: reg.Linear(p)

    for stddev in [0, 1]:
        for folds in [5, 10]:
            options = {'n': 1000, 'complexity': np.arange(20), 'stddev': stddev, 'k_folds': folds}
            test_mse = k_fold_crossvalidation(model, options)

            plt.semilogy(options['complexity'], test_mse, '-o', label = f'k = {folds} folds')
        plt.xlabel('Model Complexity (Polynomial degree)', fontsize = 12)
        plt.ylabel('Mean Squared Error', fontsize = 12)
        title = f'./results/task_c_k_fold_stddev_{stddev}.png'
        plt.legend(frameon = False)
        plt.savefig(title)
        plt.close()
