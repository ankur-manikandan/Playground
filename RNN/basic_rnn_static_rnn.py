#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:30:59 2019

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

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons,
                                   dtype=tf.float32)
outputs, state = tf.nn.static_rnn(cell, inputs=[X0, X1],
                                  dtype=tf.float32)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    s = sess.run(state, feed_dict={X0: X0_batch,
                X1: X1_batch})