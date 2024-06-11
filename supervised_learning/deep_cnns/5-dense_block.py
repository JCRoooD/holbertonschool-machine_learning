#!/usr/bin/env python3
"""Density block"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block as described in Densely Connected Convolutional
    Networks"""
    for i in range(layers):
        norm1 = K.layers.BatchNormalization()(X)
        act1 = K.layers.Activation('relu')(norm1)
        conv1 = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=1,
            padding='same',
            strides=1,
            kernel_initializer='he_normal'
        )(act1)

        norm2 = K.layers.BatchNormalization()(conv1)
        act2 = K.layers.Activation('relu')(norm2)
        conv2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            padding='same',
            strides=1,
            kernel_initializer='he_normal'
        )(act2)

        X = K.layers.concatenate([X, conv2])

        nb_filters += growth_rate

    return X, nb_filters
