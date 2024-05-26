#!/usr/bin/env python3
""" Dropout Create Layer """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """ creates a layer of a neural network using dropout
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function that should be used on the layer
        keep_prob: probability that a node will be kept
        Returns: output of the new layer
    """
    # Initialize the weights using a kernel initializer
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')

    # Create Dense layer
    dense = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer=init)

    # Apply dropout
    dropout = tf.nn.dropout(dense(prev), rate=1-keep_prob)

    return dropout
