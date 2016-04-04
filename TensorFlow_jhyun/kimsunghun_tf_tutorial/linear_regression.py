# -*- coding: utf-8 -*-
# http://hunkim.github.io/ml/
# https://www.youtube.com/watch?v=4HrSxpi3IAM&feature=youtu.be

import tensorflow as tf

# training set
x_data = [1, 2, 3]
y_data =  [1, 2, 3]

# weight and bias
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis
hypothesis = W * X + b

# Cost Function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
a = tf.Variable(0.1) # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)


# before starting, initialize the variables
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Learning
for step in xrange(1001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 10 == 0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b)
print

# Result check
print sess.run(hypothesis, feed_dict={X:5})
print sess.run(hypothesis, feed_dict={X:2.5})

sess.close()