#!/usr/bin/python
#-*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import types

# ==========================================================
#
# Dataset loading
#
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# ==========================================================
#
# Set up model
#

# 28*28 pixels -> 28^2 x(features) -> 784 features
x = tf.placeholder(tf.float32, [None, 784])
    # None -> unlimited
# To get a one y(output), there should be same number of W(weights) compares to x
# There should be 10 kinds of y,
# So W has a bimension of  784*10
W = tf.Variable(tf.zeros([784, 10]))
# b(bias) of each hypothesis for each target(label)
b = tf.Variable(tf.zeros([10]))
# Hypothesis + Softmax
y = tf.nn.softmax(tf.matmul(x, W) + b)
# y는 우리가 예측한 확률 분포이고, y_ 는 실제 분포 (우리가 입력할 one-hot 벡터)
y_ = tf.placeholder(tf.float32, [None, 10])

# Cost Function( = loss)
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy = tf.reduce_sum(y_*(-tf.log(y)))
    # ndarray로 행렬의 성분곱을 연산자 *만으로 간단히 실행 
    # labeled vector와 (-tf.log(y))의 곱으로 Cost를 정의

# Gradient Descent
# tf.train.GradientDescentOptimizer(learning_rate)
# tf.train.Optimizer.minimize(loss)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# ==========================================================
#
# Learning
#

# Session
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# Learning
# 100 개씩 뭉쳐와서 Learning 실행
# each row of batch_xs stands for each number of image(pixel data)
# each row of batch_ys stands for its label
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # stochastic training
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
# Do not close session at this point
# We just finish train the model with training set
# We are going to use this session again in testing procedure. 
# sess.close()

# ==========================================================
#
# Validation & Result
#

# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# tf.argmax(input, dimension, name=None)
    # Returns: A Tensor of type int64.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()

