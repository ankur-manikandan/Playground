#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 11:38:29 2017

@author: ankurmanikandan
"""

from itertools import product
import numpy as np
import tensorflow as tf

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

plt.style.use('ggplot')

def read_data(filename, lr, b, n_h_1, n_h_2, n_h_3, b_s, ep, count):
    
    print 'lr, b, n_h_1, n_h_2, n_h_3, b_s, ep:', lr, b, n_h_1, n_h_2, n_h_3, b_s, ep

    # read in the data
    data = pd.read_csv(filename)
    
    # Drop 'time' column. It is not going to be helpful in anyway
    data.drop(['Time'], axis=1, inplace=True)
    
    # determine the length of the dataset
    len_data = len(data)
    
    # create train and test data
    percent_test_data = 0.50  # percentage of the original data to be assigned as
    # test data
    train_data = data.iloc[:int(percent_test_data*len_data)]  # train data
    test_data = data.iloc[int(percent_test_data*len_data):]  # test data
    
    # store the 'Class' column for the train and test datasets
    train_y = pd.DataFrame(train_data['Class'])
    test_y = pd.DataFrame(test_data['Class'])
    
    # delete the 'Class' column from the train and test datasets
    train_data = train_data.drop(['Class'], axis=1)
    test_data = test_data.drop(['Class'], axis=1)
    
    # reset the test datasets index
    test_data = test_data.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)
    
    # standardize the amount column in both the test and train data
    # amount column has high values
    train_data_amount = (train_data['Amount'] - train_data.Amount.mean())/train_data.Amount.std()
    train_data['Amount'] = train_data_amount
    test_data_amount = (test_data['Amount'] - test_data.Amount.mean())/test_data.Amount.std()
    test_data['Amount'] = test_data_amount
    del train_data_amount, test_data_amount
    
    # ensure that the results are reproducible
    reset_graph()
    
    # Training Parameters
    learning_rate = lr
    beta = b
    n_input = train_data.shape[1]
    n_hidden_1 = n_h_1
    n_hidden_2 = n_h_2
    n_hidden_3 = n_h_3
    n_hidden_4 = n_hidden_2
    n_hidden_5 = n_hidden_1
    n_output = n_input
    
    batch_size = b_s
    n_epochs = ep
    nrows = train_data.shape[0]
    
    # tf Graph input (only pictures)
    X = tf.placeholder("float", [None, n_input])
    
    weights = {
        'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
        'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
        'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1)),
        'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4], stddev=0.1)),
        'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_5], stddev=0.1)),
        'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_5, n_output], stddev=0.1))
    }
    biases = {
        'encoder_b1': tf.Variable(tf.ones([n_hidden_1])/10),
        'encoder_b2': tf.Variable(tf.ones([n_hidden_2])/10),
        'encoder_b3': tf.Variable(tf.ones([n_hidden_3])/10),
        'decoder_b1': tf.Variable(tf.ones([n_hidden_4])/10),
        'decoder_b2': tf.Variable(tf.ones([n_hidden_5])/10),
        'decoder_b3': tf.Variable(tf.ones([n_output])/10)
    }
    
    # Building the encoder
    def encoder(x):
        # Encoder Hidden layer with relu activation #1
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        # Encoder Hidden layer with relu activation #2
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        # Encoder Hidden layer with relu activation #3
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                       biases['encoder_b3']))
    
        return layer_3
    
    
    # Building the decoder
    def decoder(x):
        # Decoder Hidden layer with relu activation #1
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        # Decoder Hidden layer with relu activation #2
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        # Decoder Hidden layer with relu activation #3
        layer_3 = tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                       biases['decoder_b3'])
        return layer_3
    
    # Construct model
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)
    
    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X
    
    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    reg_loss = tf.nn.l2_loss(weights['encoder_h1']) + \
               tf.nn.l2_loss(weights['encoder_h2']) + \
               tf.nn.l2_loss(weights['encoder_h3']) + \
               tf.nn.l2_loss(weights['decoder_h1']) + \
               tf.nn.l2_loss(weights['decoder_h2']) + \
               tf.nn.l2_loss(weights['decoder_h3'])
    loss = tf.reduce_mean(cost + beta * reg_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    # train and test mse
    train_mse = tf.reduce_mean(tf.pow(y_true - y_pred, 2), 1)
    test_mse = tf.reduce_mean(tf.pow(y_true - y_pred, 2), 1)
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    # Start Training
    # Start a new TF session
    with tf.Session() as sess:
        sess.run(init)
        n_batches = int(nrows/batch_size)  # number of batches the train data is split in to
        for epoch in xrange(n_epochs):
            for i in xrange(n_batches):
                x_batch = train_data[i*batch_size:(i+1)*batch_size]
                _, c = sess.run([optimizer, cost], feed_dict={X: x_batch})
            if epoch % 1 == 0:
                print("Epoch:", '%04d' % (epoch+1),
                      "cost=", "{:.9f}".format(c))
    
        print "Optimization Finished!"
        
        # determine the loss function values of the train set
        mse_train = train_mse.eval(feed_dict={X: train_data})
        
        # determine the loss function values of the test set
        mse_test = test_mse.eval(feed_dict={X: test_data})
        
    mse_df = pd.DataFrame(mse_test, columns=['test_mse'])
    mse_df['Class'] = test_y.Class.values
    
    fpr, tpr, thresholds = roc_curve(mse_df['Class'], mse_df['test_mse'])
    roc_auc = auc(fpr, tpr)
    
#    plt.figure(count+1, figsize=(10,6))
    plt.plot(fpr, tpr, color='red', label='AUC = %0.2f)' % roc_auc)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.title('ROC Autoencoder')
    plt.legend(loc="lower right")
#    plt.show()
    plt.savefig('creditCardAnalysis_v2_roc_plots/roc_'+str(count)+'.png')
    
    threshold = np.percentile(mse_train, 95)
    
    mse_df['pred'] = 0
    mse_df.loc[mse_df['test_mse'] > threshold, 'pred'] = 1
    cfm = confusion_matrix(mse_df['Class'], mse_df['pred'])
    print cfm
    
    return cfm
    
if __name__ == '__main__':
    
    data_path = 'creditcard.csv'
    
    learning_rate = [0.01, 0.03, 0.09, 0.001, 0.003, 0.009, 0.0001, 0.0003, 0.0009]
    beta = [0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]
    n_hidden_1 = [20]
    n_hidden_2 = [10]
    n_hidden_3 = [2]
    batch_size = [32]
    epochs = [1000]
    
    count = 0
    columns_list = ['count', 'learning_rate', 'beta', 'n_hidden_1',
                    'n_hidden_2', 'n_hidden_3', 'batch_size',
                    'epochs', 'tpr', 'fpr']
    params_results = pd.DataFrame(columns=columns_list)
    params_results['count'] = [0]*(len(learning_rate)*len(beta)*len(n_hidden_1)*\
                  len(n_hidden_2)*len(n_hidden_3)*len(batch_size)*len(epochs))
    
    for lr, b, n_h_1, n_h_2, n_h_3, b_s, ep in product(learning_rate, beta, n_hidden_1, n_hidden_2, n_hidden_3, batch_size, epochs):
        cfm = read_data(data_path, lr, b, n_h_1, n_h_2, n_h_3, b_s, ep, count)
        params_results.loc[count, 'count'] = count
        params_results.loc[count, 'learning_rate'] = lr
        params_results.loc[count, 'beta'] = b
        params_results.loc[count, 'n_hidden_1'] = n_h_1
        params_results.loc[count, 'n_hidden_2'] = n_h_2
        params_results.loc[count, 'n_hidden_3'] = n_h_3
        params_results.loc[count, 'batch_size'] = b_s
        params_results.loc[count, 'epochs'] = ep
        params_results.loc[count, 'tpr'] = float(cfm[1][1])/(cfm[1][0]+cfm[1][1])
        params_results.loc[count, 'fpr'] = float(cfm[0][1])/(cfm[0][0]+cfm[0][1])
        params_results.to_csv('params_results.csv', index=False)
        count += 1
