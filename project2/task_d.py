import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from model import FFNN
import activations
import loss_functions

import matplotlib.pyplot as plt

# Train FFNN on MNIST, compare with tf model

# Load data from https://www.openml.org/d/554
x, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home = './datasets')

train_samples = 60000
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_samples, test_size=10000)

# normalize data:
x_train = x_train/255.0
x_test = x_test/255.0
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

# Train tf model for comparison
input_layer = tf.keras.layers.Dense(128, activation = 'relu',
                                    kernel_initializer = 'glorot_uniform',
                                    bias_initializer = 'zeros')

output_layer = tf.keras.layers.Dense(10, activation = 'softmax',
                                    kernel_initializer = 'glorot_uniform',
                                    bias_initializer = 'zeros')

tf_model = tf.keras.Sequential([input_layer, output_layer])

lr = 1e-3
my_optimizer = tf.keras.optimizers.SGD(learning_rate = lr)
loss_func = 'categorical_crossentropy'
tf_model.compile(loss = loss_func, optimizer = my_optimizer, metrics = ['accuracy'])

# convert y_train and y_test to one_hot
y = np.zeros((len(x_train), 10)) # y_train (n_samples, 10)
y_t = np.zeros((len(x_test), 10)) # y_test

for i in range(len(x_train)): # simple, stoped
    y[i, y_train[i]] = 1

for i in range(len(x_test)):
    y_t[i, y_test[i]] = 1

epochs = 50
batch_size = 50

loss = loss_functions.SoftmaxCrossEntropy() # loss function, works with softmax
relu = activations.Relu() # hidden layer activation
softmax = activations.Softmax() # output activation

model = FFNN(loss) # build model
model.add_layer(128, relu, first_dim = x_train.shape[-1], kernel_init = 'glorot_uniform')
model.add_layer(10, softmax, kernel_init = 'glorot_uniform')

train_result, test_result = model.train(x_train, y, lr, mom = 0,
                                        epochs = epochs, bs = batch_size,
                                        val_data = (x_test, y_t),
                                        metrics = ['loss', 'accuracy'] )

fig, ax = plt.subplots(1, 2, figsize = (10, 5))

history = tf_model.fit(x_train, y, validation_data = (x_test, y_t),
           batch_size = batch_size, epochs = epochs)

tf_loss = history.history['val_loss']
tf_acc  = history.history['val_accuracy']

# plot loss, accuracy
ax[0].plot(tf_loss, label = 'Test, tf', linewidth = 0.75)
ax[1].plot(tf_acc, label = 'Test, tf', linewidth = 0.75)
for i, name in enumerate(['Crossentropy', 'Accuracy']):

    ax[i].plot(train_result[:,i], label =  'Train', linewidth = 0.75)
    ax[i].plot(test_result[:,i], label = 'Test', linewidth = 0.75)
    ax[i].legend(frameon = False)
    ax[i].set_xlabel('Epoch', fontsize = 12)
    ax[i].set_title(name, fontsize = 14)
    ax[i].legend(frameon = False)

plt.savefig('./results/task_d.png')

print('Final test accuracy:', test_result[-1, -1])
# Example run:
# Final test accuracy: 0.9269
