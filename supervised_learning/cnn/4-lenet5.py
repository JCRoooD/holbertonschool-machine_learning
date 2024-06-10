#!/usr/bin/env python3
"""LeNet-5 architecture"""
import tensorflow.compat.v1 as tf # type: ignore

tf.disable_v2_behavior()


def lenet5(x, y):
    """Builds a modified version of the
    LeNet-5 architecture using tensorflow"""
    init = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = tf.layers.conv2d(
        x,
        filters=6,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=init,
    )

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = tf.layers.conv2d(
        pool1,
        filters=16,
        kernel_size=5,
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=init,
    )

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

    # Flatten the pool2 output
    flat = tf.layers.flatten(pool2)

    # Fully connected layer with 120 nodes
    fc1 = tf.layers.dense(
        flat, units=120, activation=tf.nn.relu, kernel_initializer=init
    )

    # Fully connected layer with 84 nodes
    fc2 = tf.layers.dense(fc1, units=84,
                          activation=tf.nn.relu, kernel_initializer=init)

    # Fully connected softmax output layer with 10 nodes
    logic = tf.layers.dense(
        flat, units=120, activation=tf.nn.relu, kernel_initializers=init
    )

    # Loss
    loss = tf.losses.softmax_cross_entropy(y, logic)

    # Training operation
    train_op = tf.train.AdamOptimizer().minimize(loss)

    softmax = tf.nn.softmax(logic)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return softmax, train_op, loss, accuracy