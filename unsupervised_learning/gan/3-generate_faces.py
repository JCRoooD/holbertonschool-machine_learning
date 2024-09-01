#!/usr/bin/env python3
""" This module has the method train"""
import tensorflow as tf
from tensorflow import keras

def convolutional_GenDiscr():
    """ This method creates a convolutional generator and discriminator"""
    def get_generator():
        """ This method creates a generator"""
        inputs = keras.Input(shape=(16,))
        x = keras.layers.Dense(2048, activation='tanh')(inputs)
        x = keras.layers.Reshape((2, 2, 512))(x)
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(16, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(1, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation('tanh')(x)
        generator = keras.Model(inputs, outputs, name="generator")
        return generator

    def get_discriminator():
        inputs = keras.Input(shape=(16, 16, 1))
        x = keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1)(x)
        discriminator = keras.Model(inputs, outputs, name="discriminator")
        return discriminator

    return get_generator(), get_discriminator()

# Example usage:
gen, discr = convolutional_GenDiscr()
print(gen.summary(line_length=100))
print(discr.summary(line_length=100))