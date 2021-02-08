# -*- coding: utf-8 -*-

import os

class config(object):
    alpha = 1
    optim = 'SGD'  # Options: SGD, SGD+Momentum, Adam
    lr_decay = False
    train_epoch = 100000 # A large number with break_when_reaching_zero_error = True
    break_when_reaching_zero_error = True


    # Data loading
    num_workers = 4
    train_batch_size = 128
    test_batch_size = 128
    sharpness_train_batch_size = 128

    binary_dataset = True # This needs to be "True" when calculating volume.


    # Sharpness & Sensitivity & Volume
    sharpness_method = 'SGD' # Option: 'SGD', 'L-BFGS-B'
    sharpness_cons = False # Consecutive recording sharpness.
    sharpness_one_off = True # Compute sharpness at last epoch.
    sensitivity_cons = False # Recording input sensitivity for each epoch of training.
    sensitivity_one_off = False # Compute input sensitivity at the end of traning.
    robustness_one_off = False

    volume_one_off = True

    # General
    record = True
    attack_set_size = 0
