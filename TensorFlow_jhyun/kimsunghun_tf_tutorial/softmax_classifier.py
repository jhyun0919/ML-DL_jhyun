# -*- coding: utf-8 -*-
# http://hunkim.github.io/ml/
# https://www.youtube.com/watch?v=FiPpqSqR_1c

import tensorflow as tf
import numpy as np

# xy = np.transpose(np.loadtxt('train_for_soft.txt', unpack=True, dtype='float32'))
#
# x_data = xy[:, :3]
# y_data = xy[:, 3:]

xy = np.loadtxt('train_for_soft.txt', unpack=True, dtype='float32')

x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

print xy
print x_data
print y_data

# tf Graph input
X = tf.placeholder('float', [None, 3])  # x1, x2, bias
Y = tf.placeholder('float', [None, 3])  # A, B, and C -> 3 classes

# set model weight
W = tf.Variable(tf.zeros([3, 3]))

# Construct model
hypothesis = tf.nn.softmax(tf.matmul(X, W))

# Learning rate
learning_rate = 0.001

# Cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the Graph
with tf.Session() as sess:
    sess.run(init)
    for step in xrange(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print step, sess.run(cost, feed_dict={X: x_data, Y: y_data})
            print sess.run(W)


    # Test & one-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7]]})
    print a, sess.run(tf.argmax(a, 1))

    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4]]})
    print b, sess.run(tf.argmax(b, 1))

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0]]})
    print c, sess.run(tf.arg_max(c, 1))

    # all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
    # print all, sess.run(tf.arg_max(all, 1))
