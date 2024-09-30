#!/usr/bin/env python3
"""This modlue contains the RNNDcoder class"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    This class represents the decoder for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        class constructor
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        This method builds the decoder

        """
        context, _ = self.attention(s_prev, hidden_states)

        # Pass the previous word index through the embedding layer
        x = self.embedding(x)

        # Concatenate the context vector with x
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        # Pass the concatenated vector through the GRU layer
        output, s = self.gru(x)

        # Remove the extra axis
        output = tf.squeeze(output, axis=1)

        # Pass the GRU output through the Dense layer to predict the next word
        y = self.F(output)

        return y, s
