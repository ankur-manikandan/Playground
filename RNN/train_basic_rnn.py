#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 21:08:58 2019

@author: ankurmanikandan
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

n_time_steps = 28
n_features = 28
n_neurons = 150
n_outputs = 10

mnist = input_data.read_data_sets("/tmp/data/")
x_test = mnist.test.images.reshape((-1, n_time_steps, n_features))
y_test = mnist.test.labels

learning_rate = 0.001
n_epochs = 100
batch_size = 150

x = tf.placeholder(tf.float32, [None, n_time_steps, n_features])
y = tf.placeholder(tf.int32, [None])

cell = tf.nn.rnn_cell.BasicRNNCell(n_neurons)
outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs, activation=None)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    n_batches = mnist.train.num_examples//batch_size
    for epoch in xrange(n_epochs):
        for iteration in range(n_batches):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            x_batch = x_batch.reshape((-1, n_time_steps, n_features))
            sess.run(optimizer, feed_dict={x: x_batch,
                                           y: y_batch})
        acc_train = accuracy.eval(feed_dict={x: x_batch,
                                             y: y_batch})
        acc_test = accuracy.eval(feed_dict={x: x_test,
                                            y: y_test})
        print epoch, "Train accuracy: ", acc_train, "Test accuracy: ",\
        acc_test
            

