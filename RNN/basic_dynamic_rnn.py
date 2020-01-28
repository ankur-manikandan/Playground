#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:53:20 2019

@author: ankurmanikandan
"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

x_batch = np.random.random((8, 3)).reshape((4, 2, 3))

n_time_steps = 2
n_features = 3
n_neurons = 5

x = tf.placeholder(tf.float32, [None, n_time_steps, n_features])
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons,
                                   dtype=tf.float32)
outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    y_vals = outputs.eval(feed_dict={x: x_batch})
    states_vals = states.eval(feed_dict={x: x_batch})
