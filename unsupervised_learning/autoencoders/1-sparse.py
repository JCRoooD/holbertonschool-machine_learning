#!/usr/bin/env python3
""" Sparse autoencoder module"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """ creates an sparse autoencoder
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list of integers containing the number of nodes for each
            hidden layer in the encoder, respectively
            the hidden layers should be reversed for the decoder
        latent_dims: integer containing the dimensions of the latent space
        lambtha: the regularization parameter used for L1 regularization on the
            encoded output
        Returns: encoder, decoder, auto
            encoder: the encoder model
            decoder: the decoder model
            auto: the full autoencoder model
    """
    input_img = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(hidden_layers[0], activation='relu')(input_img)
    for hl in hidden_layers[1:]:
        encoded = keras.layers.Dense(hl, activation='relu')(encoded)
    latent = keras.layers.Dense(latent_dims, activation='relu',
                                activity_regularizer=keras.regularizers.l1(lambtha))(encoded)
    encoder = keras.Model(input_img, latent)

    input_latent = keras.Input(shape=(latent_dims,))
    decoded = keras.layers.Dense(hidden_layers[-1], activation='relu')(input_latent)
    for hl in hidden_layers[-2::-1]:
        decoded = keras.layers.Dense(hl, activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(input_latent, decoded)

    auto = keras.Model(input_img, decoder(encoder(input_img)))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
