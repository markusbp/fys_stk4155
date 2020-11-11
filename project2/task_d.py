import tensorflow as tf
import numpy as np

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
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0].reshape(28,28))
# plt.colorbar()
# plt.show()

input_layer = tf.keras.layers.Dense(128, activation = 'relu', kernel_initializer = 'random_normal', bias_initializer = 'zeros')
output_layer = tf.keras.layers.Dense(10, activation = 'softmax', kernel_initializer = 'random_normal', bias_initializer = 'zeros')
tf_model = tf.keras.Sequential([input_layer, output_layer])

lr = 1e-4
my_optimizer = tf.keras.optimizers.SGD(learning_rate = lr)
loss_func = 'categorical_crossentropy'

tf_model.compile(loss = loss_func, optimizer = my_optimizer, metrics = ['accuracy'])

y = np.zeros((len(x_train), 10))

y_t = np.zeros((len(x_test), 10))

for i in range(len(x_train)):
    y[i, y_train[i]] = 1

for i in range(len(x_test)):
    y_t[i, y_test[i]] = 1

epochs = 20 # antall ganger vi kjører gjennom hele datasettet
batch_size = 50

loss = loss_functions.SoftmaxCrossEntropy()
relu = activations.Relu()
softmax = activations.Softmax()

model = FFNN(loss)
model.add_layer(128, relu, first_dim = x_train.shape[-1])
model.add_layer(10, softmax)

train_result, test_result = model.train(x_train, y, lr, mom = 0, epochs = epochs, bs = batch_size, val_data = (x_test, y_t), metrics = ['loss', 'accuracy'] )

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize = (10, 5))

history = tf_model.fit(x_train, y, validation_data = (x_test, y_t),
           batch_size = batch_size, epochs = epochs)

tf_loss = history.history['val_loss']
tf_acc  = history.history['val_accuracy']
# loss , accuracy
ax[0].plot(tf_loss, label = 'Test, tf', linewidth = 0.75)
ax[1].plot(tf_acc, label = 'Test, tf', linewidth = 0.75)
for i, name in enumerate(['Crossentropy', 'Accuracy']):

    ax[i].plot(train_result[:,i], label =  'Train', linewidth = 0.75)
    ax[i].plot(test_result[:,i], label = 'Test', linewidth = 0.75)
    ax[i].legend(frameon = False)
    ax[i].set_xlabel('Epoch', fontsize = 12)
    ax[i].set_title(name, fontsize = 14)
    ax[i].legend(frameon = False)
    #ax[i].axis('equal')

plt.savefig('./results/task_d.png')
