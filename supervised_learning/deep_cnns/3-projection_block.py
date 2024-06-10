#!/usr/bin/env python3
"""Projection Block"""
from tensorflow import keras as K



def projection_block(A_prev, filters, s=2):
    """Builds a projection block as described in
    Deep Residual Learning for Image Recognition (2015)"""
    F11, F3, F12 = filters

    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        strides=s,
        padding='same',
        activation='linear',
        kernel_initializer=K.initializers.he_normal(seed=None)
    )(A_prev)

    conv1 = K.layers.BatchNormalization(axis=3)(conv1)
    conv1 = K.layers.Activation('relu')(conv1)

    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding='same',
        activation='linear',
        kernel_initializer=K.initializers.he_normal(seed=None)
    )(conv1)

    conv2 = K.layers.BatchNormalization(axis=3)(conv2)
    conv2 = K.layers.Activation('relu')(conv2)

    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        padding='same',
        activation='linear',
        kernel_initializer=K.initializers.he_normal(seed=None)
    )(conv2)

    conv3 = K.layers.BatchNormalization(axis=3)(conv3)

    conv4 = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        strides=s,
        padding='same',
        activation='linear',
        kernel_initializer=K.initializers.he_normal(seed=None)
    )(A_prev)

    conv4 = K.layers.BatchNormalization(axis=3)(conv4)

    output = K.layers.Add()([conv3, conv4])

    output = K.layers.Activation('relu')(output)

    return output
