#!/usr/bin/env python3
"""Identity Block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Builds an identity block as described in
    Deep Residual Learning for Image Recognition (2015)"""
    F11, F3, F12 = filters

    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(A_prev)

    conv1 = K.layers.BatchNormalization(axis=3)(conv1)
    conv1 = K.layers.Activation('relu')(conv1)

    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(conv1)

    conv2 = K.layers.BatchNormalization(axis=3)(conv2)
    conv2 = K.layers.Activation('relu')(conv2)

    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(conv2)

    conv3 = K.layers.BatchNormalization(axis=3)(conv3)
    conv3 = K.layers.Activation('relu')(conv3)

    output = K.layers.Add()([conv3, A_prev])
    output = K.layers.Activation('relu')(output)

    return output
