#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 17:14:41 2017

@author: ankurmanikandan
"""

'''
The following code details how a neural network works.

The dataset that is used is the Credit Card Fraud Dataset that is available
on Kaggle. We use the neural network architecture to build a supervised
classification algorithm. We classify fraudulent transactions (label: 1) 
from the normal transactions (label: 0).

Link to the dataset: https://www.kaggle.com/dalpozz/creditcardfraud/data
'''

import numpy as np
import pandas as pd


def _read_data(data_path):
    '''
    '''
    
    data = pd.read_csv(data_path)
    
    # drop 'Time' column
    data.drop(['Time'], axis=1, inplace=True)
    
    return data

def _train_dev_test_data(orig_data, percent_train_data,
                         percent_dev_data, len_data):
    '''
    '''
    
    # train data
    train_data = orig_data.loc[:percent_train_data*len_data]
    train_data.reset_index(drop=True, inplace=True)  # reset the index
    
    # cross val/development dataset
    dev_data = orig_data.loc[int(percent_train_data*len_data):int((
            percent_train_data+percent_dev_data)*len_data)]
    dev_data.reset_index(drop=True, inplace=True)  # reset the index
    
    # test data
    test_data = orig_data.loc[int((percent_train_data+percent_dev_data)*len_data):]
    test_data.reset_index(drop=True, inplace=True)  # reset the index
    
    return train_data, dev_data, test_data

def _init_neural_net(input_values):
    '''
    '''
    
    

def neural_network_main(data_path, input_values):
    '''
    '''
    
    # read in the data
    data = _read_data(data_path)
    
    # determine the length of the dataset
    len_data = len(data)
    
    # obtain train and test datasets
    train_data, dev_data, test_data = _train_dev_test_data(data,
                                                           input_values['percent_train_data'],
                                                           input_values['percent_dev_data'],
                                                           len_data)
    
    # initialize the neural network architecture
    _init_neural_net(input_values)
    
    # train the neural network
    
    return train_data, dev_data, test_data

if __name__ == '__main__':
    
    # the path to where the data is stored
    data_path = 'creditcard.csv'
    
    # dictionary containing the input values
    input_values = {'percent_train_data': 0.50,  # percent of data we want 
                    # to use as training data
                    'percent_dev_data': 0.25,  # percent of data we want to
                    # use as cross val/development data
                    'n_layers': 5,  # number of hidden layers in the network
                    'n_neurons': [28, 28, 28, 28, 28],  # number of neurons 
                    # in each hidden layer
                    'learning_rate': 0.001,  # learning rate
                    'regularization_param': 0.01  # regularization parameter
                    }
    
    train_data, dev_data, test_data = neural_network_main(data_path, input_values)
