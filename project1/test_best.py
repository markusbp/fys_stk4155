import numpy as np
import matplotlib.pyplot as plt
import grid_search as gs
import regression as reg

import tools
import terrain

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    model = reg.RidgeRegression(14, 1e-4)  # best model from grid search

    r, labels = terrain.get_dataset()
    x_train, x_test, y_train, y_test = train_test_split(r, labels, test_size = 0.2)

    # number of samples to go into crossvalidation
    n_train = len(x_train)

    scaler = tools.Scaler(x_train) # scale before crossvalidation
    scaled_x_train = scaler(x_train)
    scaled_x_test = scaler(x_test)

    model.fit(scaled_x_train, y_train)

    r_full, labels_full = terrain.get_full_dataset()

    preds = model.predict(scaler(r_full)) # scale, then predict

    r2 = np.mean((preds - labels_full)**2)


    fig, ax = plt.subplots(1, 2, figsize = (13, 5))
    im1 = ax[0].imshow(labels_full.reshape(100,100), aspect = 'auto', cmap = 'coolwarm')
    im2 = ax[1].imshow(preds.reshape(100,100), aspect = 'auto', cmap = 'coolwarm')
    fig.colorbar(im1, ax = ax[0])
    fig.colorbar(im2, ax = ax[1])
    ax[0].set_xlabel('x',fontsize = 12)
    ax[0].set_ylabel('y', fontsize = 12)
    ax[1].set_xlabel('x',fontsize = 12)
    ax[1].set_ylabel('y', fontsize = 12)
    plt.savefig('./results/best_performer.png')

    print('Full image MSE:', r2)

    # example run:
    # Full image MSE: 0.006834092081369142
