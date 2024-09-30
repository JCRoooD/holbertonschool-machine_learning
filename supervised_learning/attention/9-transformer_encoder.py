#!/usr/bin/env python3
"""This module conatins the Encoder class
that inherits from tensorflow.keras.layers.Layer"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """This class creates an encoder for a transformer"""
    def __init__(
            self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        """
        Class constructor
        """
        # call the parent class constructor
        super(Encoder, self).__init__()

        # set the number of blocks
        self.N = N
        # Set the dimensionality of the model
        self.dm = dm
        # set the embedding layer
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        # set the positional encoding layer
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        # set the list of encoder blocks
        self.blocks = [EncoderBlock(
            dm, h, hidden, drop_rate) for _ in range(N)]
        # set the dropout layer
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        This method builds the encoder
        """
        seq_len = x.shape[1]

        # apply the embedding layer
        x = self.embedding(x)
        # scale the embedding by the square root of the dimension
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        # add the positional encoding
        x += self.positional_encoding[:seq_len]

        # apply the dropout layer
        x = self.dropout(x, training=training)

        # call each block in the encoder
        for block in self.blocks:
            x = block(x, training, mask)

        return x
