#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 00:04:54 2019

@author: ankurmanikandan
"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

X0_batch = np.random.random((4, 3))    
X1_batch = np.random.random((4, 3))

n_features = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_features])
X1 = tf.placeholder(tf.float32, [None, n_features])

Wx = tf.Variable(tf.random_normal([n_features, n_neurons]), 
                 dtype=tf.float32)
Wy = tf.Variable(tf.random_normal([n_neurons, n_neurons]), 
                 dtype=tf.float32)
b = tf.Variable(tf.zeros([1, n_neurons]), dtype=tf.float32)

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(X1, Wx) + tf.matmul(Y0, Wy) + b)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0:X0_batch,
                              X1: X1_batch})
    
