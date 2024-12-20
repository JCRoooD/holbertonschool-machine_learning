#!/usr/bin/env python3
"""Inception Network"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the inception network as described in
    Going Deeper with Convolutions (2014)"""
    X = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        padding='same',
        strides=2,
        activation='relu',
        kernel_initializer='he_normal'
    )(X)

    pool1 = K.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(conv1)

    conv2 = K.layers.Conv2D(
        filters=64,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(pool1)

    conv3 = K.layers.Conv2D(
        filters=192,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(conv2)

    pool2 = K.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(conv3)

    inception1 = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    inception2 = inception_block(inception1, [128, 128, 192, 32, 96, 64])

    pool3 = K.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(inception2)

    inception3 = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    inception4 = inception_block(inception3, [160, 112, 224, 24, 64, 64])
    inception5 = inception_block(inception4, [128, 128, 256, 24, 64, 64])
    inception6 = inception_block(inception5, [112, 144, 288, 32, 64, 64])
    inception7 = inception_block(inception6, [256, 160, 320, 32, 128, 128])

    pool4 = K.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(inception7)

    inception8 = inception_block(pool4, [256, 160, 320, 32, 128, 128])

    inception9 = inception_block(inception8, [384, 192, 384, 48, 128, 128])

    avg_pool = K.layers.AveragePooling2D(
        pool_size=7,
        strides=1
    )(inception9)

    dropout = K.layers.Dropout(0.4)(avg_pool)

    output = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer='he_normal'
    )(dropout)

    model = K.Model(inputs=X, outputs=output)

    return model
