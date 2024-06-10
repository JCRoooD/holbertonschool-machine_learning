#!/usr/bin/env python3
"""ResNet-50"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)"""
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

    projection1 = projection_block(pool1, [64, 64, 256], s=1)
    identity1 = identity_block(projection1, [64, 64, 256])
    identity2 = identity_block(identity1, [64, 64, 256])

    projection2 = projection_block(identity2, [128, 128, 512])
    identity3 = identity_block(projection2, [128, 128, 512])
    identity4 = identity_block(identity3, [128, 128, 512])
    identity5 = identity_block(identity4, [128, 128, 512])

    projection3 = projection_block(identity5, [256, 256, 1024])
    identity6 = identity_block(projection3, [256, 256, 1024])
    identity7 = identity_block(identity6, [256, 256, 1024])
    identity8 = identity_block(identity7, [256, 256, 1024])
    identity9 = identity_block(identity8, [256, 256, 1024])
    identity10 = identity_block(identity9, [256, 256, 1024])

    projection4 = projection_block(identity10, [512, 512, 2048])
    identity11 = identity_block(projection4, [512, 512, 2048])
    identity12 = identity_block(identity11, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(
        pool_size=7,
        strides=1
    )(identity12)

    output = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer='he_normal'
    )(avg_pool)

    model = K.Model(inputs=X, outputs=output)

    return model
