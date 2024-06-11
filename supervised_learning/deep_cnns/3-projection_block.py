#!/usr/bin/env python3
"""Projection Block"""
from tensorflow import keras as K



def projection_block(A_prev, filters, s=2):
    """Builds a projection block as described in
    Deep Residual Learning for Image Recognition (2015)"""
    F11, F3, F12 = filters

    x_shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        strides=s,
        padding='same',
        kernel_initializer='he_normal'
    )(A_prev)
    x_shortcut = K.layers.BatchNormalization()(x_shortcut)

    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        strides=s,
        padding='same',
        kernel_initializer='he_normal'
    )(A_prev)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer='he_normal'
    )(X)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal'
    )(X)
    X = K.layers.BatchNormalization()(X)

    X = K.layers.Add()([X, x_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
