import numpy as np
import matplotlib.pyplot as plt

import tools
import regression as reg
import statistics as stats
import franke_function as frank

from sklearn.model_selection import train_test_split


def task_a():

    p = 5 # Degree of (2D) polynomial to use

    n = 1000 # number of data points
    sig = 1.96 # significance leve, corresponds to 95% CI

    stddev = [0, 1] # Data noise

    for std, marker in zip(stddev, ['.', '*']):
        r, labels = frank.get_dataset(n, stddev = std)

        x_train, x_test, y_train, y_test = train_test_split(r, labels, test_size = 0.2)

        input_scaler = tools.Scaler(x_train)
        scaled_train_x = input_scaler(x_train)
        scaled_test_x = input_scaler(x_test)

        model = reg.Linear(p)

        model.fit(scaled_train_x, y_train)

        train_preds = model.predict(scaled_train_x)
        test_preds = model.predict(scaled_test_x)
        # Compute statistics
        train_mse = stats.mse(train_preds, y_train)
        test_mse = stats.mse(test_preds, y_test)
        r2 = stats.r2_score(test_preds, y_test)

        # find variance of least squares estimator beta; (x^Tx)^-1*sigma^2
        # estimate error as 1/(N-p-1)*sum of residuals
        sigma2 = 1/(n-p-1)*test_mse*len(y_test) # Test MSE = 1/n_test*sum of residuals
        X = model.design_matrix(scaled_test_x) # get design matrix
        var_beta = np.diag(sigma2*np.linalg.inv(X.T@X))

        errorbars = sig*np.sqrt(var_beta) # 95 % confidence interval

        plt.errorbar(np.arange(len(model.beta_)), model.beta_, yerr = errorbars,
                     fmt = marker, label = f'$S_D$ = {std}', alpha = 0.5)
        plt.ylabel('Coefficient Value', fontsize = 12)
        plt.xlabel('Coefficient No.', fontsize = 12)
        print(f'Training loss {train_mse}, test loss {test_mse}')
        print(f'Test R2 Score {r2}\n')

    plt.legend(frameon = False)
    plt.savefig('./results/task_a')

if __name__ == '__main__':
    task_a()

    # run example
    '''
    Training loss 0.0016740177505400733, test loss 0.0021402492653484957
    Test R2 Score 0.9781600158154178

    Training loss 0.8749968363131034, test loss 0.9967125779891868
    Test R2 Score 0.07195101635998213

    '''
