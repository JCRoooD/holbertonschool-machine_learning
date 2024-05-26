#!/usr/bin/env python3
""" L2 Regularization Cost """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ creates a tensorflow layer that includes L2 regularization
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function that should be used on the layer
        lambtha: L2 regularization parameter
        Returns: output of the new layer
    """
    # Initialize the weights using a kernel regularizer
    initializer = tf.keras.initializers.VarianceScaling(mode="fan_avg")
    regularizer = tf.keras.regularizers.l2(lambtha)

    # Create the layer
    layer = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer)

    # Connect the new layer to the previous layer
    output = layer(prev)

    return output
