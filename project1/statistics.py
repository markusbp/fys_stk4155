import numpy as np

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def r2_score(y_pred, y_true):
    y_mean = np.mean(y_true)
    r2 = 1 - np.sum((y_true - y_pred)**2)/np.sum((y_true - y_mean)**2)
    return r2
