import numpy as np
import model

from model import FFNN
import activations
import loss_functions

mse = loss_functions.MSE()
relu = activations.Relu()

model = FFNN(mse)
model.add_layer(5, relu, first_dim = 3)

model.add_layer(7, relu)


x0 = np.random.uniform(-1, 1, (1000, 10, 3))

y0 = np.random.uniform(-1, 1, (1000, 10, 7))

model.train(x0, y0, 1e-5, 0, epochs = 1000)
