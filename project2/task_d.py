import tensorflow as tf
import tensorflow_datasets as tfds


from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from model import FFNN
import activations
import loss_functions

# Load data from https://www.openml.org/d/554
x, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home = './datasets')

train_samples = 60000
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_samples, test_size=10000)

# normalize data:
x_train = x_train/255.0
x_test = x_test/255.0
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')
# Tensorflow model


import matplotlib.pyplot as plt
plt.imshow(x_train[0].reshape(28,28))
plt.colorbar()
plt.show()


input_layer = tf.keras.layers.Dense(20, activation = 'relu')
output_layer = tf.keras.layers.Dense(10, activation = 'softmax')
model = tf.keras.Sequential([input_layer, output_layer])

lr = 1e-4
my_optimizer = tf.keras.optimizers.SGD(learning_rate = lr)
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

#model.build()
model.compile(loss = loss_func, optimizer = my_optimizer, metrics = ['accuracy'])


epochs = 10 # antall ganger vi kjører gjennom hele datasettet
batch_size = 50
model.fit(x_train, y_train, validation_data = (x_test, y_test),
          batch_size = batch_size, epochs = epochs)

# cross_entropy = ---
# relu = ...
# softmax = ...
loss = loss_functions.MSE()
relu = activations.Relu()
softmax = activations.Softmax()

model = FFNN(loss)
model.add_layer(100, relu, first_dim = 3)
model.add_layer(10, softmax)
model.train(x_train, y_train, lr, momentum = 0, epochs = epochs, bs = batch_size)
