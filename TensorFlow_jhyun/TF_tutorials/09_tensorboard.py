# -*- coding: utf-8 -*-

import tensorflow as tf
import input_data


#############################################################################################
# function define
#############################################################################################

def init_weights(shape, name):
    """
    초기화 함수
    :param shape: 반환할 variable 의 shape
    :param name: 반환할 variable 의 name
    :return: tf.Variable
    """
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


# This network is the same as the previous one except with an extra hidden layer + dropout
def model(X, w_h, w_h2, w_o, dropout_rate_input, dropout_rate_hidden):
    # Add layer name scopes for better graph visualization
    with tf.name_scope("layer1"):
        # dropout 을 통해 선별적으로 activation node 택함
        X = tf.nn.dropout(X, dropout_rate_input)
        # matmul 과 relu 를 이용하여 연산 실행
        h = tf.nn.relu(tf.matmul(X, w_h))
    with tf.name_scope("layer2"):
        h = tf.nn.dropout(h, dropout_rate_hidden)
        h2 = tf.nn.relu(tf.matmul(h, w_h2))
    with tf.name_scope("layer3"):
        h2 = tf.nn.dropout(h2, dropout_rate_hidden)
        return tf.matmul(h2, w_o)


#############################################################################################
# Load MNIST Data
#############################################################################################

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

#############################################################################################
# Make Session Model
#############################################################################################

# 상수들이 들아갈 공간 placeholder 를 적당한 shape 로 설정하여 확보
X = tf.placeholder("float", [None, 784], name="X")
Y = tf.placeholder("float", [None, 10], name="Y")

# Session 에 사용될 변수, variable 들을 미리 정의한 함수를 사용하여 확보 & 초기화
w_h = init_weights([784, 625], "w_h")
w_h2 = init_weights([625, 625], "w_h2")
w_o = init_weights([625, 10], "w_o")

# Add histogram summaries for weights
tf.histogram_summary("w_h_summ", w_h)
tf.histogram_summary("w_h2_summ", w_h2)
tf.histogram_summary("w_o_summ", w_o)

# dropout_rate 설정
dropout_rate = tf.placeholder("float", name="dropout_rate")

# Make model 
py_x = model(X, w_h, w_h2, w_o, dropout_rate, dropout_rate)

# Cost scope
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    # Add scalar summary for cost
    tf.scalar_summary("cost", cost)

# Accuracy scope
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1))  # Count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float"))  # Cast boolean to float to average
    # Add scalar summary for accuracy
    tf.scalar_summary("accuracy", acc_op)

# Run Session
with tf.Session() as sess:
    # create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    writer = tf.train.SummaryWriter("./logs/nn_logs", sess.graph)  # for 0.8
    merged = tf.merge_all_summaries()

    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          dropout_rate: 0.8})
        summary, acc = sess.run([merged, acc_op], feed_dict={X: teX, Y: teY,
                                                             dropout_rate: 1.0})
        writer.add_summary(summary, i)  # Write summary
        print(i, acc)  # Report the accuracy
