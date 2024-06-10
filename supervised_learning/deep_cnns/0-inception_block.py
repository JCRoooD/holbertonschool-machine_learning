#!/usr/bin/env python3
"""Inception Block"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Builds an inception block as described in
    Going Deeper with Convolutions (2014)"""
    F1, F3R, F3, F5R, F5, FPP = filters

    conv1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(A_prev)

    conv3r = K.layers.Conv2D(
        filters=F3R,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(A_prev)

    conv3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(conv3r)

    conv5r = K.layers.Conv2D(
        filters=F5R,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(A_prev)

    conv5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(conv5r)

    pool_proj = K.layers.MaxPool2D(
        pool_size=3,
        strides=1,
        padding='same'
    )(A_prev)

    convPP = K.layers.Conv2D(
        filters=FPP,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(pool_proj)

    return K.layers.concatenate([conv1, conv3, conv5, convPP])
