#!/usr/bin/env python3
"""create_train_op"""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """Creates the training operation for the network"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
