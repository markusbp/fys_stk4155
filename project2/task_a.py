import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor

import terrain
import regression as reg
from gradient_descent import GD



data, labels = terrain.get_dataset()

r, r_test, y, y_test = train_test_split(data, labels)


ols_model = reg.Linear(p = 14)
sgd_model = reg.Linear(p = 14)
sk_model = SGDRegressor(fit_intercept=False, alpha = 0, eta0 = 1e-5)

pipe = PolynomialFeatures(degree=14)

ols_model.fit(r, y)

epochs = 100
batch_size = 50
trainer = GD(lr0 = 1e-4, momentum = 0.1) # gradient descent
sgd_model.fit_sgd(r, y, batch_size, epochs, trainer)

for i in range(epochs):
    inds = np.random.choice(len(r), batch_size)
    r0 = r[inds]
    y0 = y[inds]
    r0 = pipe.fit_transform(r0)
    sk_model.partial_fit(r0, y0)
    err_sk = np.mean((y_test - sk_model.predict(pipe.fit_transform(r_test)))**2)
    plt.plot(i, err_sk, 'o')
plt.show()

err_ols = np.mean((y_test - ols_model.predict(r_test))**2)
err_sgd = np.mean((y_test - sgd_model.predict(r_test))**2)



print('ols', err_ols, 'sgd', err_sgd, 'sk', err_sk)
