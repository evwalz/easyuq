# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his 
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

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

import isodisreg
from isodisreg import idr
import pandas as pd



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
            @param tau          regularization value 
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
        outputs = Dense(y_train_normalized.shape[1], kernel_regularizer=l2(reg))(inter)
        model = Model(inputs, outputs)

        model.compile(loss='mean_squared_error', optimizer='adam')

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
            @return m       The predictive mean for the test target variables.

        """

        X_test = np.array(X_test, ndmin = 2)
        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)
        model = self.model
        test_pred = model.predict(X_test, batch_size=500, verbose=0)
        test_pred = test_pred * self.std_y_train + self.mean_y_train
        return test_pred.squeeze()


    def predictIDR(self, X_test, X_train, y_test, y_train):
        
        """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data    
            @return m       The predictive mean for the test target variables.

        """
        X_test = np.array(X_test, ndmin = 2)
        X_train = np.array(X_train, ndmin = 2)
 
        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)
        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
                 np.full(X_train.shape, self.std_X_train)

        model = self.model

        test_pred = model.predict(X_test, batch_size=500, verbose=0)
        test_pred = test_pred * self.std_y_train + self.mean_y_train
        test_pred_pd = pd.DataFrame({"fore": test_pred.squeeze()}, columns=["fore"])
        
        rmse_standard_pred = np.mean((y_test.squeeze() - test_pred.squeeze()) ** 2.)**0.5

        train_pred = model.predict(X_train, batch_size=500, verbose=0)
        train_pred = train_pred * self.std_y_train + self.mean_y_train
        train_pred_pd = pd.DataFrame({"fore": train_pred.squeeze()}, columns=["fore"])
 
        # Fit IDR and make predictions
        fit_idr_train = idr(y_train, train_pred_pd)
        idr_pred = fit_idr_train.predict(test_pred_pd)
        crps_idr = idr_pred.crps(y_test)
        return rmse_standard_pred, idr_pred, np.mean(crps_idr)        

