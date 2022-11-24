# This code is based on the code by Yarin Gal.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his 
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

# This file contains code to train EasyUQ on UCI datasets using the following algorithm:
# 1. Create 20 random splits of the training-test dataset.
# 2. For each split:
# 3.   Create a validation (val) set taking 20% of the training set.
# 4.   Get best hyperparameters: regularization and batch size by training on (train-val) set and testing on val set.
# 5.   Train a network on the entire training set with the best pair of hyperparameters.
# 6.   Get the performance on the test set.
# 7. Report the averaged performance (CRPS and log-likelihood) on all 20 splits.

import math
import numpy as np
import argparse
import sys
import tensorflow

parser=argparse.ArgumentParser()

parser.add_argument('--dir', '-d', required=True, help='Name of the UCI Dataset directory. Eg: bostonHousing')
parser.add_argument('--epochx','-e', default=400, type=int, help='Multiplier for the number of epochs for training.')
parser.add_argument('--hidden', '-nh', default=2, type=int, help='Number of hidden layers for the neural net')

args=parser.parse_args()

data_directory = args.dir
epochs_multiplier = args.epochx
num_hidden_layers = args.hidden

import net_ensemble
from functions_deepens import 

# We delete previous results

from subprocess import call


_RESULTS_TEST_LL = "./UCI_Datasets/" + data_directory + "/results/test_ll_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_CRPSS = "./UCI_Datasets/" + data_directory + "/results/test_crpss_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_CRPSIDR = "./UCI_Datasets/" + data_directory + "/results/test_crpsidr_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_REG = "./UCI_Datasets/" + data_directory + "/results/test_reg_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_BATCH = "./UCI_Datasets/" + data_directory + "/results/test_batch_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_H = "./UCI_Datasets/" + data_directory + "/results/test_bw_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_DF = "./UCI_Datasets/" + data_directory + "/results/test_df_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_FLAG = "./UCI_Datasets/" + data_directory + "/results/test_flag_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_LOG = "./UCI_Datasets/" + data_directory + "/results/log_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"


_DATA_DIRECTORY_PATH = "./UCI_Datasets/" + data_directory + "/data/"
_DATA_FILE = _DATA_DIRECTORY_PATH + "data.txt"
_HIDDEN_UNITS_FILE = _DATA_DIRECTORY_PATH + "n_hidden.txt"
_EPOCHS_FILE = _DATA_DIRECTORY_PATH + "n_epochs.txt"
_INDEX_FEATURES_FILE = _DATA_DIRECTORY_PATH + "index_features.txt"
_INDEX_TARGET_FILE = _DATA_DIRECTORY_PATH + "index_target.txt"
_N_SPLITS_FILE = _DATA_DIRECTORY_PATH + "n_splits.txt"

# Added for EasyUQ
_BATCH_FILE = "./UCI_Datasets/" + data_directory + "/data/batch_sizes.txt"
_REG_VALUES_FILE = "./UCI_Datasets/" + data_directory + "/data/reg_values.txt" 

def _get_index_train_test_path(split_num, train = True):
    """
       Method to generate the path containing the training/test split for the given
       split number (generally from 1 to 20).
       @param split_num      Split number for which the data has to be generated
       @param train          Is true if the data is training data. Else false.
       @return path          Path of the file containing the requried data
    """
    if train:
        return _DATA_DIRECTORY_PATH + "index_train_" + str(split_num) + ".txt"
    else:
        return _DATA_DIRECTORY_PATH + "index_test_" + str(split_num) + ".txt" 


print ("Removing existing result files...")
call(["rm", _RESULTS_TEST_LL])
call(["rm", _RESULTS_TEST_CRPSS])
call(["rm", _RESULTS_TEST_CRPSIDR])
call(["rm", _RESULTS_TEST_BATCH])
call(["rm", _RESULTS_TEST_REG])
call(["rm", _RESULTS_TEST_H])
call(["rm", _RESULTS_TEST_DF])
call(["rm", _RESULTS_TEST_FLAG])
call(["rm", _RESULTS_TEST_LOG])
print ("Result files removed.")

# We fix the random seed

np.random.seed(1)
tensorflow.random.set_seed(2)

print ("Loading data and other hyperparameters...")
# We load the data

data = np.loadtxt(_DATA_FILE)

# We load the number of hidden units

n_hidden = np.loadtxt(_HIDDEN_UNITS_FILE).tolist()

# We load the number of training epocs

n_epochs = np.loadtxt(_EPOCHS_FILE).tolist()

# We load the indexes for the features and for the target

index_features = np.loadtxt(_INDEX_FEATURES_FILE)
index_target = np.loadtxt(_INDEX_TARGET_FILE)

X = data[ : , [int(i) for i in index_features.tolist()] ]
y = data[ : , int(index_target.tolist()) ]

# We iterate over the training test splits

n_splits = np.loadtxt(_N_SPLITS_FILE)
print ("Done.")

crps_scdf, crps_idrs, lls = [], [], []
for split in range(int(n_splits)):

    # We load the indexes of the training and test sets
    print ('Loading file: ' + _get_index_train_test_path(split, train=True))
    print ('Loading file: ' + _get_index_train_test_path(split, train=False))
    index_train = np.loadtxt(_get_index_train_test_path(split, train=True))
    index_test = np.loadtxt(_get_index_train_test_path(split, train=False))

    X_train = X[ [int(i) for i in index_train.tolist()] ]
    y_train = y[ [int(i) for i in index_train.tolist()] ]
    
    X_test = X[ [int(i) for i in index_test.tolist()] ]
    y_test = y[ [int(i) for i in index_test.tolist()] ]

    X_train_original = X_train
    y_train_original = y_train
    num_training_examples = int(0.8 * X_train.shape[0])
    X_validation = X_train[num_training_examples:, :]
    y_validation = y_train[num_training_examples:]
    X_train = X_train[0:num_training_examples, :]
    y_train = y_train[0:num_training_examples]
    
    # Printing the size of the training, validation and test sets
    print ('Number of training examples: ' + str(X_train.shape[0]))
    print ('Number of validation examples: ' + str(X_validation.shape[0]))
    print ('Number of test examples: ' + str(X_test.shape[0]))
    print ('Number of train_original examples: ' + str(X_train_original.shape[0]))

    # List of hyperparameters which we will try out using grid-search
    reg_values = np.loadtxt(_REG_VALUES_FILE).tolist()
    batch_vals = np.loadtxt(_BATCH_FILE).tolist()
    batch_vals = [int(x) for x in batch_vals]

    # We perform grid-search to select the best hyperparameters based on the highest log-likelihood value
    best_network = None
    best_ll = float('inf')
    best_reg = 0
    best_batch = 32
    best_h = 1
    best_free_degree = 2
    test_flag = 0
    
    for bs in batch_vals:
        for reg in reg_values:
            network = net_easyuq.net(X_train.copy(), y_train.copy(), ([int(n_hidden)] * num_hidden_layers),
                              normalize=True, n_epochs=int(n_epochs * epochs_multiplier), reg=reg, batch_size = bs)
        
            rmse, idr_preds_validation, crps_idr = network.predictIDR(X_validation.copy(), X_train.copy(),
                                                                  y_validation.copy(), y_train.copy())

            fitted_ll, fitted_h, fitted_df, flag = optimize_paras(idr_preds_validation, y_validation, y_train)
            
            if (fitted_ll < best_ll):
                best_ll = fitted_ll
                best_network = network
                best_reg = reg
                best_batch = bs
                best_free_degree = fitted_df
                best_h = fitted_h
                test_flag = flag

    # Storing test results
    best_network = net_easyuq.net(X_train_original, y_train_original, ([int(n_hidden)] * num_hidden_layers),
                               normalize=True, n_epochs=int(n_epochs * epochs_multiplier), reg=best_reg, batch_size = best_batch)

    
    rmse, idr_preds_test, crps_idr = best_network.predictIDR(X_test, X_train_original, y_test, y_train_original)

    ll_test = llscore(idr_preds_test, y_test, best_h, best_free_degree)
    crps_test = smooth_crps(idr_preds_test, y_test, best_h, best_free_degree)

    
    
    with open(_RESULTS_TEST_LL, "a") as myfile:
        myfile.write(repr(ll_test) + '\n')

    with open(_RESULTS_TEST_CRPSS, "a") as myfile:
        myfile.write(repr(crps_test) + '\n')

    with open(_RESULTS_TEST_CRPSIDR, "a") as myfile:
        myfile.write(repr(crps_idr) + '\n')

    with open(_RESULTS_TEST_REG, "a") as myfile:
        myfile.write(repr(best_reg) + '\n')

    with open(_RESULTS_TEST_BATCH, "a") as myfile:
        myfile.write(repr(best_batch) + '\n')

    with open(_RESULTS_TEST_H, "a") as myfile:
        myfile.write(repr(best_h) + '\n')

    with open(_RESULTS_TEST_DF, "a") as myfile:
        myfile.write(repr(best_free_degree) + '\n')

    with open(_RESULTS_TEST_FLAG, "a") as myfile:
        myfile.write(repr(test_flag) + '\n')
        
    print ("Tests on split " + str(split) + " complete.")
    crps_scdf += [crps_test]
    crps_idrs += [crps_idr]
    lls += [ll_test]

with open(_RESULTS_TEST_LOG, "a") as myfile:
    myfile.write('CRPS idr %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(crps_idrs), np.std(crps_idrs), np.std(crps_idrs)/math.sqrt(n_splits),
        np.percentile(crps_idrs, 50), np.percentile(crps_idrs, 25), np.percentile(crps_idrs, 75)))
    myfile.write('CRPS smooth%f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(crps_scdf), np.std(crps_scdf), np.std(crps_scdf)/math.sqrt(n_splits),
        np.percentile(crps_scdf, 50), np.percentile(crps_scdf, 25), np.percentile(crps_scdf, 75)))
    myfile.write('lls %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(lls), np.std(lls), np.std(lls)/math.sqrt(n_splits), 
        np.percentile(lls, 50), np.percentile(lls, 25), np.percentile(lls, 75)))
