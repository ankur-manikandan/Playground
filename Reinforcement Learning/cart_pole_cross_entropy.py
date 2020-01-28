#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 00:15:34 2019

@author: ankurmanikandan

Code taken from here: https://gist.github.com/rplis/1fe40a72656e90132ab29624f7ef5fa7
"""

import gym
import numpy as np
import tensorflow as tf
import pandas as pd

env = gym.make('CartPole-v1')  
OBSERVATION_SPACE = env.observation_space.shape[0]  # 4
ACTION_SPACE = env.action_space.n  # 2

LEARNING_RATE = 0.01
observationsPlaceholder = tf.placeholder(
    shape=[None, OBSERVATION_SPACE], dtype=tf.float64)
actionsPlaceholder = tf.placeholder(
    shape=[None, ACTION_SPACE], dtype=tf.float64)
weights = tf.Variable(tf.random_uniform(
    shape=[OBSERVATION_SPACE, ACTION_SPACE], dtype=tf.float64))
model = tf.matmul(observationsPlaceholder, weights)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=model, labels=actionsPlaceholder))
train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

EPISODES_BATCH = 100  # number of episodes as the baseline for selecting the elites
NB_TRAININGS = 100  # training iterations
QUANTILE = 0.9  # boundary for elite selection

for _ in range(NB_TRAININGS):
    df = pd.DataFrame()
    for _ in range(EPISODES_BATCH):
        observation = env.reset()
        actions = []
        observations = []
        observations.append(observation)
        rewards = 0
        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)
            print reward, done
            observations.append(observation)
            rewards += reward
            if action == 1:
                actions.append([0, 1])
            else:
                actions.append([1, 0])
        df = pd.concat([df, pd.concat([pd.DataFrame(observations),
                                       pd.DataFrame(actions),
                                       pd.DataFrame(
                                           [rewards for _ in range(int(rewards))])
                                       ], axis=1)])
    dfElite = df[df.iloc[:, -1] > df.iloc[:, -1].quantile(QUANTILE)]
    _, l = sess.run([train, loss], feed_dict={observationsPlaceholder: dfElite.iloc[:, 0:4].values,
                                              actionsPlaceholder: dfElite.iloc[:, 4:6].values})
    print "Loss: ", l, "Rewards: ", rewards

observation = env.reset()
rewards = 0
done = False
while not done:
    prediction = sess.run(
        model, feed_dict={observationsPlaceholder: observation.reshape(1, -1)})
    action = np.argmax(prediction)
    observation, reward, done, _ = env.step(action)
    rewards += reward
print("The result is:", rewards)
sess.close()