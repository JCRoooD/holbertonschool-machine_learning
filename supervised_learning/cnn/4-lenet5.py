#!/usr/bin/env python3
"""LeNet-5 architecture"""
import tensorflow.compat.v1 as tf 

def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow

    Args:
        x (tf.placeholder): Placeholder for the input images, shape (m, 28, 28, 1)
        y (tf.placeholder): Placeholder for the one-hot labels, shape (m, 10)

    Returns:
        output (tf.Tensor): Softmax activated output
        train_op (tf.Operation): Adam optimizer training operation
        loss (tf.Tensor): Loss of the network
        accuracy (tf.Tensor): Accuracy of the network
    """
    # He normal initializer
    init = tf.keras.initializers.VarianceScaling(scale=2.0)

    # First Convolutional layer
    W1 = tf.Variable(init([5, 5, 1, 6]))
    b1 = tf.Variable(tf.zeros([6]))
    conv1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, b1)
    conv1 = tf.nn.relu(conv1)
    # First Max pooling layer
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Second Convolutional layer
    W2 = tf.Variable(init([5, 5, 6, 16]))
    b2 = tf.Variable(tf.zeros([16]))
    conv2 = tf.nn.conv2d(pool1, W2, strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, b2)
    conv2 = tf.nn.relu(conv2)
    # Second Max pooling layer
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten the output
    flat = tf.layers.flatten(pool2)

    # First Fully connected layer
    fc1 = tf.layers.dense(flat, units=120, activation=tf.nn.relu, kernel_initializer=init)
    # Second Fully connected layer
    fc2 = tf.layers.dense(fc1, units=84, activation=tf.nn.relu, kernel_initializer=init)

    # Output layer
    output = tf.layers.dense(fc2, units=10, activation=tf.nn.softmax, kernel_initializer=init)
    
    # Loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output))
    # Training operation
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Accuracy calculation
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return output, train_op, loss, accuracy
