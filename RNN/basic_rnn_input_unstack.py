#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:23:29 2019

@author: ankurmanikandan
"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

x_batch = np.random.random((8, 3)).reshape(4, 2, 3)

n_time_steps = 2
n_features = 3
n_neurons = 5

x = tf.placeholder(tf.float32, [None, n_time_steps, n_features])
x_seqs = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons,
                                   dtype=tf.float32)
outputs_seqs, state = tf.nn.static_rnn(cell, inputs=x_seqs,
                                  dtype=tf.float32)
outputs = tf.transpose(tf.stack(outputs_seqs), perm=[1, 0, 2])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    y_val = outputs.eval(feed_dict={x: x_batch})


