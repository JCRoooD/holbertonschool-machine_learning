#!/usr/bin/env python3
""" Convolutional autoencoder module"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """ creates a convolutional autoencoder
        input_dims: tuple (int, int, int) containing the dimensions of the model input
        filters: list containing the number of filters for each convolutional layer in the encoder, respectively
            the filters should be reversed for the decoder
        latent_dims: tuple (int, int, int) containing the dimensions of the latent space representation
        Returns: encoder, decoder, auto
            encoder: the encoder model
            decoder: the decoder model
            auto: the full autoencoder model
    """

    encoder_inputs = keras.Input(shape=input_dims)


    # Create the encoder layers
    for idx, units in enumerate(filters):
        # Add convolutional layers with the relu activation function
        layer = keras.layers.Conv2D(
            filters=units,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
        )

        if idx == 0:
            # If it is the first layer, set the input
            outputs = layer(encoder_inputs)

        else:
            # If it is not the first layer, set the output of the previous layer
            outputs = layer(outputs)

        layer = keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=None, padding="same"
        )

        outputs = layer(outputs)

    encoder = keras.models.Model(inputs=encoder_inputs, outputs=outputs)

    decoder_inputs = keras.Input(shape=latent_dims)

    for idx, units in enumerate(reversed(filters)):

        if idx != len(filters) - 1:
            layer = keras.layers.Conv2D(
                filters=units,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                activation="relu",
            )

            if idx == 0:
                outputs = layer(decoder_inputs)

            else:
                outputs = layer(outputs)

        else:
            layer = keras.layers.Conv2D(
                filters=units,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
            )

            outputs = layer(outputs)


        layer = keras.layers.UpSampling2D(size=(2, 2))

        outputs = layer(outputs)


    layer = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="sigmoid",
    )

    outputs = layer(outputs)

    decoder = keras.models.Model(inputs=decoder_inputs, outputs=outputs)

    # Define the full autoencoder model
    outputs = encoder(encoder_inputs)
    outputs = decoder(outputs)
    auto = keras.models.Model(inputs=encoder_inputs, outputs=outputs)

    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
