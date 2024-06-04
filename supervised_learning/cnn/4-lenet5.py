#!/usr/bin/env python3
"""LeNet-5 architecture"""
import tensorflow.compat.v1 as tf 


def lenet5(x, y):
    """ Builds a modified version of the LeNet-5 architecture using tensorflow

    Args:
        x is a tf.placeholder of shape (m, 28, 28, 1) containing the input images
            for the network
            m is the number of images
        y is a tf.placeholder of shape (m, 10) containing the one-hot labels for
            the network

    Returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization (with default
            hyperparameters)
        a tensor for the loss of the network
        a tensor for the accuracy of the network
    """
    init = tf.contrib.layers.variance_scaling_initializer()

    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                             activation=tf.nn.relu, kernel_initializer=init)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                             activation=tf.nn.relu, kernel_initializer=init)(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flatten = tf.layers.Flatten()(pool2)

    fc1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                         kernel_initializer=init)(flatten)
    fc2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                         kernel_initializer=init)(fc1)
    output = tf.layers.Dense(units=10, kernel_initializer=init)(fc2)

    y_pred = tf.nn.softmax(output)
    loss = tf.losses.softmax_cross_entropy(y, output)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return y_pred, train_op, loss, accuracy
