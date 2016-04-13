# -*- coding: utf-8 -*-
# http://hunkim.github.io/ml/
# https://www.youtube.com/watch?v=FiPpqSqR_1c

import tensorflow as tf

X = tf.placeholder(tf.float32, name='X-input')
Y = tf.placeholder(tf.float32, name='Y-input')

W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name='Weight1')
W2 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name='Weight1')

b1 = tf.Variable(tf.zeros([2]), name='Bias1')
b2 = tf.Variable(tf.zeros([1]), name='Bias2')


# Our hypothesis
with tf.name_scope('layer2') as scope:
    L2 = tf.sigmoid(tf.matmul(X, W1)+ b1)

with tf.name_scope('layer3') as scope:
    hypothesis = tf.sigmoid(tf.matmul(L2, W2)+ b1)

# Cost function
with tf.name_scope('cost') as scope:
    cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
    cost_sum = tf.scalar_summary('cost', cost)

# Minimize
with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(cost)


# Histogram
w1_hist = tf.histogram_summary('weight1', W1)
w2_hist = tf.histogram_summary('weight2', W2)
b1_hist = tf.histogram_summary('biases1', b1)
b2_hist = tf.histogram_summary('biases2', b2)
y_hist = tf.histogram_summary('y', Y)

# Launch the graph
with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('./logs/xor_logs', sess.graph_def)

# Fit the line
for step in xrange(20000):
    summary, _ =sess.run([merged, train], feed_dict={X:x_data, Y:y_data})
    writer.add_summary(summary, step)

    # sess.run(train, feed_dict={X:x_data, Y:y_data})
    # if step % 2000 == 0:
    #     summary = sess.run(merged, feed_dict={X:x_data, Y:y_data})
    #     writer.add_summary(summary, step)

