#!/usr/bin/env python3
"""Inception Network"""
from tensorflow import keras as K
inception = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the inception network as described in
    Going Deeper with Convolutions (2014)"""
    X = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(X)

    pool1 = K.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(conv1)

    conv2r = K.layers.Conv2D(
        filters=64,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(pool1)

    conv2 = K.layers.Conv2D(
        filters=192,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(conv2r)

    pool2 = K.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(conv2)

    inception3a = inception(pool2, [64, 96, 128, 16, 32, 32])
    inception3b = inception(inception3a, [128, 128, 192, 32, 96, 64])

    pool3 = K.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(inception3b)

    inception4a = inception(pool3, [192, 96, 208, 16, 48, 64])
    inception4b = inception(inception4a, [160, 112, 224, 24, 64, 64])
    inception4c = inception(inception4b, [128, 128, 256, 24, 64, 64])
    inception4d = inception(inception4c, [112, 144, 288, 32, 64, 64])
    inception4e = inception(inception4d, [256, 160, 320, 32, 128, 128])

    pool4 = K.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(inception4e)

    inception5a = inception(pool4, [256, 160, 320,
                                    32, 128, 128])
    inception5b = inception(inception5a, [384, 192, 384,
                                           48, 128, 128])

    avg_pool = K.layers.AveragePooling2D(
        pool_size=7,
        strides=1
    )(inception5b)

    dropout = K.layers.Dropout(0.4)(avg_pool)

    out = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer='he_normal'
    )(dropout)

    model = K.models.Model(inputs=X, outputs=out)

    return model
