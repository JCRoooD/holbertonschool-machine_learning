#!/usr/bin/env python3
""" 
that builds a neural network with the Keras library:
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ 
    Returns: the keras model
    """
    inputs = K.Input(shape=(nx,))
    L2 = K.regularizers.l2(lambtha)
    model = K.layers.Dense(layers[0], activation=activations[0], kernel_regularizer=L2)(inputs)
    for i in range(1, len(layers)):
        model = K.layers.Dropout(1 - keep_prob)(model)
        model = K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=L2)(model)
    return K.Model(inputs=inputs, outputs=model)
