#!/usr/bin/env python3
"""Module for create_layer function"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    """Create layer function"""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer,
                            name="layer")

    return layer(prev)
