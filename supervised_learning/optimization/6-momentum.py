#!/usr/bin/env python3
"""Momentum optimization algorithm"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """creates the training operation for a neural network in tensorflow
    using the gradient descent with momentum optimization algorithm:
    loss is the loss of the network
    alpha is the learning rate
    beta1 is the momentum weight
    Returns: the momentum optimization operation"""
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
