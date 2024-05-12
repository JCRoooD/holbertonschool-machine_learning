#!/usr/bin/env python3
"""forward_prop"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network"""
    create_layer = __import__('1-create_layer').create_layer
    A = x
    for i in range(len(layer_sizes)):
        A = create_layer(A, layer_sizes[i], activations[i])
    return A
