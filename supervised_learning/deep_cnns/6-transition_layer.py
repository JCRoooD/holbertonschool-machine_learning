#!/usr/bin/env python3
"""transition layer"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """builds a transition layer"""
    init = K.initializers.he_normal()
    batch_norm = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch_norm)
    conv = K.layers.Conv2D(int(nb_filters * compression),
                           kernel_size=(1, 1),
                           padding='same',
                           kernel_initializer=init)(activation)
    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                         strides=(2, 2))(conv)
    return avg_pool, int(nb_filters * compression)
