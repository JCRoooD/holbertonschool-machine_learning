#!/usr/bin/env python3
""" Variational autoencoder module"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ creates a variational autoencoder
        input_dims: tuple (int) containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden layer in the encoder, respectively
            the hidden layers should be reversed for the decoder
        latent_dims: tuple (int) containing the dimensions of the latent space representation
        Returns: encoder, decoder, auto
            encoder: the encoder model
            decoder: the decoder model
            auto: the full autoencoder model
    """

    encoder_inputs = keras.Input(shape=(input_dims,))

    for idx, units in enumerate(hidden_layers):
        # Add dense layers with the relu activation function
        layer = keras.layers.Dense(units=units, activation="relu")

        if idx == 0:
            # If it is the first layer, set the input
            outputs = layer(encoder_inputs)

        else:
            # If it is not the first layer, set the
            # output of the previous layer
            outputs = layer(outputs)

    layer = keras.layers.Dense(units=latent_dims)

    mean = layer(outputs)

    layer = keras.layers.Dense(units=latent_dims)

    log_variation = layer(outputs)

    def sampling(args):
        """This function samples from the mean and log variation
        Args:
            args: list containing the mean and log variation
        Returns: sampled tensor
        """
        # Get the mean and log variation from the arguments
        mean, log_variation = args

        epsilon = keras.backend.random_normal(shape=keras.backend.shape(mean))

        return mean + keras.backend.exp(log_variation * 0.5) * epsilon

    # Use a keras layer to wrap the sampling function
    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [mean, log_variation]
    )

    encoder = keras.models.Model(
        inputs=encoder_inputs, outputs=[z, mean, log_variation]
    )

    # Define the decoder model
    decoder_inputs = keras.Input(shape=(latent_dims,))
    for idx, units in enumerate(reversed(hidden_layers)):
        # Create a Dense layer with relu activation
        layer = keras.layers.Dense(units=units, activation="relu")

        if idx == 0:
            # if it is the first layer, set the input
            outputs = layer(decoder_inputs)

        else:
            outputs = layer(outputs)

    layer = keras.layers.Dense(units=input_dims, activation="sigmoid")

    outputs = layer(outputs)

    decoder = keras.models.Model(inputs=decoder_inputs, outputs=outputs)

    # Create the full autoencoder model
    outputs = encoder(encoder_inputs)

    outputs = decoder(outputs[0])

    auto = keras.models.Model(inputs=encoder_inputs, outputs=outputs)

    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
