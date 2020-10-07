import numpy as np

class Scaler(object):
    # scale input data
    def __init__(self, data):
        self.mu = np.mean(data)
        self.sig = np.std(data)

    def __call__(self, unscaled_data):
        scaled_data = (unscaled_data - self.mu)/self.sig
        return scaled_data

def bootstrap_with_replacement(x,y):
    # Simple bootstrap, simply resample with replacement
    inds = np.random.choice(len(x), size = len(x), replace = True)
    return x[inds], y[inds]

# def bootstrap_dataset(x, y, train_fraction = 0.8): # better bootstrap ?
#     # Leave one out bootstrap
#     n_samples = len(x)
#     n_train = int(n_samples*train_percentage)
#     test_mask = np.ones(n_samples, dtype = np.bool)
#     train_inds = np.sort(np.random.choice(n_samples, size = n_train, replace = False))
#     test_mask[train_inds] = False
#     return split_data(x, y, train_inds, test_mask)

def split_data(x, y, train_fraction = 0.8):
    # split data into train/test
    n_samples = len(x)
    n_train = int(n_samples*train_fraction)

    train_inds = np.sort(np.random.choice(n_samples, size = n_train, replace = False))
    test_mask = np.ones(n_samples, dtype = np.bool)
    test_mask[train_inds] = False

    x_train = x[train_inds]
    y_train = y[train_inds]
    x_test = x[test_mask]
    y_test = y[test_mask]
    return x_train, y_train, x_test, y_test
