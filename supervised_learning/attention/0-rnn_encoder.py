#!/usr/bin/env python3
"""module contains the RNnENcoder class"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """This class represents an encoder for a transformer
        and inherits from tf.keras.layers.Layer"""

    def __init__(self, vocab, embedding, units, batch):
        """
        class constructor
        """
        super(RNNEncoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab, embedding)

        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.batch = batch
        self.units = units

    def initialize_hidden_state(self):
        """Initializes the hidden states for the RNN cell to a tensor of zeros

        Returns:
            tf.Tensor: a tensor of shape (self.batch, self.units) containing
                the initialized hidden states
        """
        return tf.zeros((self.batch, self.gru.units))

    def call(self, x, initial):
        """This method builds the encoder

        Args:
            x (tf.Tensor): a tensor of shape (batch, input_seq_len) containing
                the input to the encoder
            initial (tf.Tensor): a tensor of shape (batch, units) containing
                the initial hidden state

        Returns:
            tf.Tensor, tf.Tensor: the outputs of the encoder and the last
                hidden state
        """
        x = self.embedding(x)

        outputs, hidden = self.gru(x, initial)
        return outputs, hidden
