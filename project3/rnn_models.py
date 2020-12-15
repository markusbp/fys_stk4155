import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class BaseLineRNN(tf.keras.Model):
    def __init__(self, options):
        # baseline RNN, options contains all model parameters
        super().__init__()
        self.options = options
        #
        self.rnn_layer = tf.keras.layers.SimpleRNN(options.nodes, return_sequences = True)
        self.output_layer = tf.keras.layers.Dense(options.out_nodes, activation = 'relu')
        self.start_state = self.add_weight(name = 'init_state', shape = (1, options.nodes))

        self.bs = options.batch_size
        self.ts = options.timesteps


        self.act_l2 = tf.keras.layers.ActivityRegularization(l2 = 1e-4)


    def call(self, inputs, training = True):

        r = inputs[1]

        initial_state = self.start_state*tf.ones((self.options.batch_size, 1))
        self.rnn_states = self.rnn_layer(inputs[0], initial_state = [initial_state])

        self.act_l2(self.rnn_states)

        self.outputs = self.output_layer(self.rnn_states)

        ta = tf.keras.activations.softmax(self.outputs*self.options.beta, axis = 1)
        pa = tf.keras.activations.softmax(self.outputs*self.options.beta, axis = 2)

        expanded_r = tf.reshape(r, (self.bs, self.ts, 1, 2))
        expanded_ta = tf.reshape(ta, (self.bs, self.ts, -1, 1))
        expanded_pa = tf.reshape(pa, (self.bs, self.ts, -1, 1))

        weighted_centers = expanded_r*expanded_ta
        self.expected_centers = tf.reduce_sum(weighted_centers, axis = 1)

        centers = tf.reshape(self.expected_centers, (self.bs, 1, -1, 2))
        weighted_positions = expanded_pa*centers
        expected_position = tf.reduce_sum(weighted_positions, axis = -2)

        return expected_position

class EgoRNN(tf.keras.Model):
    def __init__(self, options):
        super().__init__()
        self.options = options

        activation = tf.keras.layers.Activation(options.activation)
        self.output_layer = tf.keras.layers.Dense(options.out_nodes, activation = 'relu')

        random_normal = tf.keras.initializers.RandomNormal(0, 0.001)
        rec_reg = tf.keras.regularizers.l2(options.l2)
        self.rnn_layer = tf.keras.layers.SimpleRNN(options.nodes, activation = activation,
                                                   kernel_initializer = random_normal,
                                                   recurrent_initializer = 'identity',
                                                   recurrent_regularizer = rec_reg,
                                                   return_sequences = True)
        self.start_state = self.add_weight(name = 'init_state', shape = (1, options.nodes))

        if self.options.dropout_rate != 0:
            self.dropout = tf.keras.layers.Dropout(options.dropout_rate)

        self.bs = options.batch_size
        self.ts = options.timesteps

        #self.act_l2 = tf.keras.layers.ActivityRegularization(l2 = 1e-4)

    def call(self, inputs, training = True):

        r = inputs[1]

        initial_state = self.start_state*tf.ones((self.options.batch_size, 1))
        self.rnn_states = self.rnn_layer(inputs[0], initial_state = [initial_state])

        #self.act_l2(self.rnn_states)

        if self.options.dropout_rate != 0:
            dropped = self.output_layer(self.rnn_states)
            self.outputs = self.dropout(dropped, training = training)
        else:
            self.outputs = self.output_layer(self.rnn_states)

        ta = tf.keras.activations.softmax(self.outputs*self.options.beta, axis = 1)
        pa = tf.keras.activations.softmax(self.outputs*self.options.beta, axis = 2)

        expanded_r = tf.reshape(r, (self.bs, self.ts, 1, 2))
        expanded_ta = tf.reshape(ta, (self.bs, self.ts, -1, 1))
        expanded_pa = tf.reshape(pa, (self.bs, self.ts, -1, 1))

        weighted_centers = expanded_r*expanded_ta
        self.expected_centers = tf.reduce_sum(weighted_centers, axis = 1)

        centers = tf.reshape(self.expected_centers, (self.bs, 1, -1, 2))
        weighted_positions = expanded_pa*centers
        expected_position = tf.reduce_sum(weighted_positions, axis = -2)

        return expected_position


class RNNcell(tf.keras.layers.Layer):
    def __init__(self, options):
        self.units = options.nodes
        self.state_size = self.units
        self.options = options
        super().__init__()

    def build(self, input_shape):

        self.activation = tf.keras.layers.Activation(self.options.activation)

        n_speed = 20
        n_hd = 50
        vmax = 0.5
        centers = tf.convert_to_tensor(np.linspace(0, vmax, n_speed, dtype = 'float32'))
        self.speed_centers = tf.reshape(centers, (1, -1))
        self.speed_sd =  vmax/n_speed

        hd_centers = np.linspace(0, 2*np.pi, n_hd, dtype = 'float32')[None,:]
        hd_sd = 2*np.pi # gives approximate max of 1
        self.hd = tfp.distributions.VonMises(hd_centers, hd_sd)

        random_normal = tf.keras.initializers.RandomNormal(0, 0.001)

        self.speed_input = tf.keras.layers.Dense(self.units,
                            kernel_initializer = random_normal, use_bias = False)
        self.hd_input = tf.keras.layers.Dense(self.units,
                            kernel_initializer = random_normal, use_bias = False)

        rec_reg = tf.keras.regularizers.l2(self.options.l2)
        self.recurrent = tf.keras.layers.Dense(self.units,
                                               kernel_initializer = 'identity',
                                               kernel_regularizer = rec_reg)
        self.built = True

    def speed_gaussian(self,r):
        dr = tf.expand_dims(r, axis = -1)
        exponent = (dr-self.speed_centers)**2
        return tf.exp(-0.5/self.speed_sd**2*exponent)

    def call(self, inputs, states):
        speed = self.speed_gaussian(inputs[:,0])
        hd = self.hd.prob(tf.expand_dims(inputs[:,1], axis = -1))
        u = self.recurrent(states[0]) + self.hd_input(hd) + self.speed_input(speed)
        h = self.activation(u)
        return h, [h]

class RBFRNN(tf.keras.Model):
    def __init__(self, options):
        super().__init__()
        self.options = options

        rnn_cell = RNNcell(options)
        self.rnn_layer = tf.keras.layers.RNN(rnn_cell, return_sequences = True)

        if self.options.dropout_rate != 0:
            self.dropout = tf.keras.layers.Dropout(options.dropout_rate)


        self.output_layer = tf.keras.layers.Dense(options.out_nodes, activation = 'relu')

        self.start_state = self.add_weight(name = 'init_state', shape = (1, options.nodes))

        self.bs = options.batch_size
        self.ts = options.timesteps

    def call(self, inputs, training = True):

        r = inputs[1]

        initial_state = self.start_state*tf.ones((self.options.batch_size, 1))
        self.rnn_states = self.rnn_layer(inputs[0], initial_state = [initial_state])

        if self.options.dropout_rate != 0:
            dropped = self.output_layer(self.rnn_states)
            self.outputs = self.dropout(dropped, training = training)
        else:
            self.outputs = self.output_layer(self.rnn_states)

        ta = tf.keras.activations.softmax(self.outputs*self.options.beta, axis = 1)
        pa = tf.keras.activations.softmax(self.outputs*self.options.beta, axis = 2)

        expanded_r = tf.reshape(r, (self.bs, self.ts, 1, 2))
        expanded_ta = tf.reshape(ta, (self.bs, self.ts, -1, 1))
        expanded_pa = tf.reshape(pa, (self.bs, self.ts, -1, 1))

        weighted_centers = expanded_r*expanded_ta
        self.expected_centers = tf.reduce_sum(weighted_centers, axis = 1)

        centers = tf.reshape(self.expected_centers, (self.bs, 1, -1, 2))
        weighted_positions = expanded_pa*centers
        expected_position = tf.reduce_sum(weighted_positions, axis = -2)

        return expected_position
