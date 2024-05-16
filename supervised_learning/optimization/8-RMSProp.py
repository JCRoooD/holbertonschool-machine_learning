#!/usr/bin/env python3
"""set up RMSProp optimization algorithm"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """creates the training operation for a neural network in tensorflow
    using the RMSProp optimization algorithm:
    loss is the loss of the network
    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero
    Returns: the RMSProp optimization operation"""
    optimize = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                           rho=beta2,
                                           epsilon=epsilon)
    return optimize
