import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.utils import resample

import franke_function as frank
import regression as reg
from gradient_descent import GD

import seaborn as sns

def bootstrap_mse(model, x_test, y_test, n_bootstraps = 100):
    err = 0
    for i in range(n_bootstraps):
        b_x, b_y = resample(x_test, y_test)
        err += np.mean((model.predict(b_x) - b_y)**2)
    return err/n_bootstraps

def bootstrap_r2(model, x_test, y_test, n_bootstraps = 100):
    r2 = 0
    for i in range(n_bootstraps):
        b_x, b_y = resample(x_test, y_test)
        y_bar = np.mean(b_y)
        pred = model.predict(b_x)
        r2 += 1 - np.sum( (pred - b_y)**2 )/np.sum((b_y - y_bar)**2)
    return r2/n_bootstraps

def search_ols(r_train, y_train, r_test, y_test, epochs = 1):

    bs = [10, 50, 100, len(r_train)]
    learning_rates = np.geomspace(1e-4, 0.1, 4)
    static = [True, False]

    error = np.zeros((2, len(bs), len(learning_rates)))
    r2 = np.zeros((2, len(bs), len(learning_rates)))

    fig, ax = plt.subplots(2, 2, figsize = (10, 10))

    for h, decay in enumerate(static):
        for i, batch_size in enumerate(bs):
            for j, lr in enumerate(learning_rates):

                sgd_model = reg.Linear(p = 14)

                trainer = GD(lr0 = lr, decay = decay) # gradient descent
                sgd_model.fit_sgd(r_train, y_train, batch_size, epochs, trainer)

                error[h,i,j] = bootstrap_mse(sgd_model, r_test, y_test)
                r2[h,i,j] = bootstrap_r2(sgd_model, r_test, y_test)

        if decay == False:
            title = 'Constant LR'
        else:
            title = 'LR Decay'

        sns.heatmap(error[h], annot=True, ax=ax[h,0], cmap="magma",
                    xticklabels = learning_rates, yticklabels = bs)
        ax[h,0].set_title("Test MSE " + title)
        ax[h,0].set_xlabel("Learning Rate", fontsize = 12)
        ax[h,0].set_ylabel("Batch Size", fontsize = 12)

        sns.heatmap(r2[h], annot=True, ax=ax[h,1], cmap="magma",
                    xticklabels = learning_rates, yticklabels = bs)

        ax[h,1].set_title("Test R2 Score "  + title)
        ax[h,1].set_xlabel("Learning Rate", fontsize = 12)
        ax[h,1].set_ylabel("Batch Size", fontsize = 12)

    plt.tight_layout()
    plt.savefig('./results/task_a_ols.png')

    ols_model = reg.Linear(p = 14)
    ols_model.fit(r_train, y_train)

    ols_err = bootstrap_mse(ols_model, r_test, y_test)
    ols_r2 = bootstrap_r2(ols_model, r_test, y_test)

    sk_model = SGDRegressor(fit_intercept=False, alpha = 0, eta0 = lr)
    pipe = PolynomialFeatures(degree=14)

    # Fit Sklearn model
    for i in range(epochs):
        inds = np.random.choice(len(r_train), batch_size)
        r0 = r_train[inds]
        y0 = y_train[inds]
        r0 = pipe.fit_transform(r0)
        sk_model.partial_fit(r0, y0)

    r_test = pipe.fit_transform(r_test) # sklearn design matrix
    sk_error = bootstrap_mse(sk_model, r_test, y_test)
    sk_r2 = bootstrap_r2(sk_model, r_test, y_test)

    print('SKlearn model MSE:', sk_error, '. SKlearn model R2:', sk_r2)
    print('OLS model MSE:', ols_err, '. OLS model R2:', ols_r2)


def search_ridge(r_train, y_train, r_test, y_test, epochs = 1):

    static = True
    batch_size = 10
    lams = np.geomspace(1e-5, 0.01, 4)
    learning_rates = np.geomspace(1e-4, 0.1, 4)

    error = np.zeros((len(lams), len(learning_rates)))
    r2 = np.zeros((len(lams), len(learning_rates)))

    fig, ax = plt.subplots(1, 2, figsize = (10, 5))

    for i, lam in enumerate(lams):
        for j, lr in enumerate(learning_rates):

            sgd_model = reg.RidgeRegression(14, lam)

            trainer = GD(lr0 = lr, decay = False) # gradient descent
            sgd_model.fit_sgd(r_train, y_train, batch_size, epochs, trainer)

            error[i,j] = bootstrap_mse(sgd_model, r_test, y_test)
            r2[i,j] = bootstrap_r2(sgd_model, r_test, y_test)

    sns.heatmap(error, annot=True, ax=ax[0], cmap="magma",
                xticklabels = learning_rates, yticklabels = lams)
    ax[0].set_title("Test MSE")
    ax[0].set_xlabel("Learning Rate", fontsize = 12)
    ax[0].set_ylabel("$\lambda$", fontsize = 12)

    sns.heatmap(r2, annot=True, ax=ax[1], cmap="magma",
                xticklabels = learning_rates, yticklabels = lams)

    ax[1].set_title("Test R2 Score")
    ax[1].set_xlabel("Learning Rate", fontsize = 12)
    ax[1].set_ylabel("$\lambda$", fontsize = 12)

    plt.tight_layout()
    plt.savefig('./results/task_a_ridge.png')

if __name__ == '__main__':
    samples = 1000

    data, labels = frank.get_dataset(samples)
    r_train, r_test, y_train, y_test = train_test_split(data, labels)

    mean = np.mean(r_train)
    r_train = r_train - mean
    r_test = r_test - mean

    search_ols(r_train, y_train, r_test, y_test, epochs = 100)
    search_ridge(r_train, y_train, r_test, y_test, epochs = 100)

    #Run example:
    # SKlearn model MSE: 0.04958971687144914 . SKlearn model R2: 0.6063338850674247
    # OLS model MSE: 0.0463499927777239 . OLS model R2: 0.6348382786709478
