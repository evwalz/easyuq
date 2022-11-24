# Gaussian Layer code from: https://medium.com/@albertoarrigoni/paper-review-code-deep-ensembles-nips-2017-c5859070b8ce 
# Note for protein-tertiary-structure and year use 100 and not 50

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np

from keras.regularizers import l2
from keras import Input
from keras.layers import Dropout
from keras.layers import Dense
from keras import Model

import time


class GaussianLayer(Layer):        
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(GaussianLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel_1 = self.add_weight(name='kernel_1', 
                                      shape=(50, self.output_dim),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.kernel_2 = self.add_weight(name='kernel_2', 
                                      shape=(50, self.output_dim),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.bias_1 = self.add_weight(name='bias_1',
                                    shape=(self.output_dim, ),
                                    initializer=glorot_normal(),
                                    trainable=True)
        self.bias_2 = self.add_weight(name='bias_2',
                                    shape=(self.output_dim, ),
                                    initializer=glorot_normal(),
                                    trainable=True)
        super(GaussianLayer, self).build(input_shape)     
    def call(self, x):
        output_mu  = K.dot(x, self.kernel_1) + self.bias_1
        output_sig = K.dot(x, self.kernel_2) + self.bias_2
        output_sig_pos = K.log(1 + K.exp(output_sig)) + 1e-06  
        return [output_mu, output_sig_pos]    
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]




class net:

    def __init__(self, X_train, y_train, n_hidden, n_epochs = 40,
        normalize = False, reg = 0, batch_size = 32):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
            @param reg          regularization value 
            @param batch_size   batch size in neural network
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        if normalize:
            self.std_X_train = np.std(X_train, 0)
            self.std_X_train[ self.std_X_train == 0 ] = 1
            self.mean_X_train = np.mean(X_train, 0)
        else:
            self.std_X_train = np.ones(X_train.shape[ 1 ])
            self.mean_X_train = np.zeros(X_train.shape[ 1 ])

        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
            np.full(X_train.shape, self.std_X_train)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
        y_train_normalized = np.array(y_train_normalized, ndmin = 2).T
        
        # We construct the network
        N = X_train.shape[0]


        inputs = Input(shape=(X_train.shape[1],))
        inter = Dense(n_hidden[0], activation='relu', kernel_regularizer=l2(reg))(inputs)
        for i in range(len(n_hidden) - 1):
            inter = Dense(n_hidden[i+1], activation='relu', kernel_regularizer=l2(reg))(inter)
        mu, var = GaussianLayer(1, name = 'main_output')(inter)
        combined_outputs = Concatenate()([mu, var])
        model = Model(inputs, combined_outputs)

        def log_loss(true, pred):
            loss = K.mean( 0.5*(K.log(pred[:, 1:])) + 0.5 * K.square(true - pred[:, 0:1]) / pred[:, 1:], -1) + 1e-6
            return loss
        model.compile(loss=log_loss, optimizer='adam')


        # We iterate the learning process
        start_time = time.time()
        model.fit(X_train, y_train_normalized, batch_size=batch_size, epochs=n_epochs, verbose=0)
        self.model = model
        self.running_time = time.time() - start_time

        # We are done!

    def predict(self, X_test):

        """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data    
            @return m       The predictive mean and sigma for the test target variables.

        """

        X_test = np.array(X_test, ndmin = 2)
        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)
        model = self.model
        
        standard_pred = model.predict(X_test, batch_size=500, verbose=1)
        standard_pred = standard_pred * [self.std_y_train, self.std_y_train**2] + [self.mean_y_train, 0]

        return standard_pred
