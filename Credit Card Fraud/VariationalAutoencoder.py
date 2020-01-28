#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:18:28 2017

@author: ankurmanikandan
"""

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

# In[]:

# read in the data
filename = 'creditcard.csv'
data = pd.read_csv(filename)

# Drop 'time' column. It is not going to be helpful in anyway
data.drop(['Time'], axis=1, inplace=True)

# Split the data into fraudluent and non-fradulent transactions
nonfraud_data = data[data.Class == 0]
fraud_data = data[data.Class == 1]

# The train set will only contain a portion of the non-fraudulent transacitons
train_data = nonfraud_data.iloc[:150000]
train_data_class = train_data.Class  # Store the train 'Class' variable in another array
train_data = train_data.drop(['Class'], axis=1)  # Drop the 'Class' column from the test data

# The test set will be a combnation of both fraudulent and non-fraudulent transactions
test_data = nonfraud_data.iloc[150000:]
test_data = pd.concat([test_data, fraud_data])
test_data_class = test_data.Class  # store the 'Class' variable in another array
test_data.drop(['Class'], axis=1, inplace=True)  # Need to drop the 'Class' column in the test data

# In[]:

reset_graph()

# Shape of the data
ncols_data = train_data.shape[1]
nrows_data = train_data.shape[0]

# Set the parameters of the variational autoencoder
learning_rate = 0.001
n_hidden_1 = 20
n_hidden_2 = 10
ndim_z = 5
n_hidden_3 = n_hidden_2
n_hidden_4 = n_hidden_1
n_output = ncols_data

batch_size = 64
n_epochs = 50

# In[]:

# Create a placeholder for X
X = tf.placeholder("float", shape=[None, ncols_data])
X_sigmoid = tf.nn.sigmoid(X)

# Define the weights for each layer
weights = {'w_input_hidden1': tf.Variable(tf.truncated_normal([ncols_data, n_hidden_1], stddev=0.01)),
           'w_hidden1_hidden2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.01)),
           'w_hidden2_mean': tf.Variable(tf.truncated_normal([n_hidden_2, ndim_z], stddev=0.01)),
           'w_hidden2_var': tf.Variable(tf.truncated_normal([n_hidden_2, ndim_z], stddev=0.01)),
           'w_z_hidden3': tf.Variable(tf.truncated_normal([ndim_z, n_hidden_3], stddev=0.01)),
           'w_hidden3_hidden4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4], stddev=0.01)),
           'w_hidden4_output': tf.Variable(tf.truncated_normal([n_hidden_4, ncols_data], stddev=0.01))
           }

# Define the biases for each layer
biases = {'b_input_hidden1': tf.zeros([n_hidden_1]),
          'b_hidden1_hidden2': tf.zeros([n_hidden_2]),
          'b_hidden2_mean': tf.zeros([ndim_z]),
          'b_hidden2_var': tf.zeros([ndim_z]),
          'b_z_hidden3': tf.zeros([n_hidden_3]),
          'b_hidden3_hidden4': tf.zeros([n_hidden_4]),
          'b_hidden4_output': tf.zeros([ncols_data])
          }

# Define the encoder network
h1 = tf.nn.elu(tf.matmul(X_sigmoid, weights['w_input_hidden1']) + biases['b_input_hidden1'])
h2 = tf.nn.elu(tf.matmul(h1, weights['w_hidden1_hidden2']) + biases['b_hidden1_hidden2'])
latent_mean = tf.matmul(h2, weights['w_hidden2_mean']) + biases['b_hidden2_mean']
latent_var = tf.matmul(h2, weights['w_hidden2_var']) + biases['b_hidden2_var']
latent_sigma = tf.exp(0.5 * latent_var)
# Before we are able to sample z from the mean and var we need to generate some noise using
# a standard normal distribution
eps = tf.random_normal(tf.shape(latent_sigma), 0, 1, dtype=tf.float32)
latent_z = latent_mean + latent_sigma * eps
h3 = tf.nn.elu(tf.matmul(latent_z, weights['w_z_hidden3']) + biases['b_z_hidden3'])
h4 = tf.nn.elu(tf.matmul(h3, weights['w_hidden3_hidden4']) + biases['b_hidden3_hidden4'])
output = tf.matmul(h4, weights['w_hidden4_output']) + biases['b_hidden4_output']

# Reconstruction loss
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=X_sigmoid, logits=output), 1)
# Latent loss
latent_loss = 0.5 * tf.reduce_sum(tf.exp(latent_var) + tf.square(latent_mean) - 1 - latent_var, 1)
# Total losss
cost = tf.reduce_mean(recon_loss + latent_loss)

# Train loss
train_cost = recon_loss + latent_loss
# Test loss
test_cost = recon_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    n_batches = nrows_data/batch_size
    for epoch in xrange(n_epochs):
        for i in xrange(n_batches):
            x_batch = train_data.iloc[i*batch_size : (1+i)*batch_size]
            _, c = sess.run([optimizer, cost], feed_dict={X: x_batch})
        # Display logs per epoch step
        if epoch % 1 == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))
    
    print "Optimization Finished!"
        
    # determine the loss function values of the test set
    test_loss = test_cost.eval(feed_dict={X: test_data})
    
    # determine the loss function values of the test set
    train_loss = train_cost.eval(feed_dict={X: train_data})
    
    print "Done!"
    
# In[]:
    
fpr, tpr, thresholds = roc_curve(test_data_class, test_loss)
roc_auc = auc(fpr, tpr)  # determine the auc (best balance of tpr and fpr)

# Plot of fpr vs. tpr
plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, color='red', label='AUC = %0.2f)' % roc_auc)
plt.xlim((0,1))
plt.ylim((0,1))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title('ROC Autoencoder')
plt.legend(loc="lower right")
plt.show()

# In[]:

threshold = np.percentile(train_loss, 75)

# create the predicted class column in the dataframe
mse_df = np.zeros(len(test_loss))
# If the test mse is greater than the threshold then assign the label 1.
mse_df[mse_df >= threshold] = 1

# Since we have the actual class labels for the test set we can determine -
# - how well the autoencoder has performed. 
# We use the confusion matrix
cfm = confusion_matrix(test_data_class, mse_df)
print 'Confusion Matrix: '
print cfm
print '\n'
print 'True Positive Rate: ', float(cfm[1][1])/(cfm[1][0] + cfm[1][1])
print 'False Positive Rate: ', float(cfm[0][1])/(cfm[0][0] + cfm[0][1])
print 'False Negative Rate: ', float(cfm[1][0])/(cfm[1][0] + cfm[1][1])
    
