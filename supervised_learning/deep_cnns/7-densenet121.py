#!/usr/bin/env python3
"""desnet121"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate, compression):
    """builds the DenseNet-121 architecture 
    as described in Densely Connected Convolutional Networks"""
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)

    norm1 = K.layers.BatchNormalization()(X)
    act1 = K.layers.Activation('relu')(norm1)
    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        padding='same',
        strides=2,
        kernel_initializer=init
    )(act1)

    pool1 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(conv1)

    Y, nb_filters = dense_block(pool1, 64, growth_rate, 6)
    Y, nb_filters = transition_layer(Y, nb_filters, compression)

    Y, nb_filters = dense_block(Y, nb_filters, growth_rate, 12)
    Y, nb_filters = transition_layer(Y, nb_filters, compression)

    Y, nb_filters = dense_block(Y, nb_filters, growth_rate, 24)
    Y, nb_filters = transition_layer(Y, nb_filters, compression)

    Y, nb_filters = dense_block(Y, nb_filters, growth_rate, 16)

    avg_pool = K.layers.AveragePooling2D(
        pool_size=7,
        strides=None,
        padding='same'
    )(Y)

    output = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=init
    )(avg_pool)

    model = K.models.Model(inputs=X, outputs=output)

    return model
