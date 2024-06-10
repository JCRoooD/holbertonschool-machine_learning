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
        kernel_initializer='he_normal'
    )(A_prev)

    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(conv1)

    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(conv2)

    return K.layers.add([conv3, A_prev])
