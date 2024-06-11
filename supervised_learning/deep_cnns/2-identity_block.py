#!/usr/bin/env python3
"""Identity Block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Builds an identity block as described in
    Deep Residual Learning for Image Recognition (2015)"""
    F11, F3, F12 = filters

    #input value for the shortcut
    X_shortcut = A_prev

    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        padding='same',
        kernel_initializer='he_normal'
    )(A_prev)

    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding='same',
        kernel_initializer='he_normal'
    )(X)

    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        padding='same',
        kernel_initializer='he_normal'
    )(X)

    X = K.layers.BatchNormalization()(X)

    X = K.layers.Add()([X, X_shortcut])

    X = K.layers.Activation('relu')(X)

    return X
