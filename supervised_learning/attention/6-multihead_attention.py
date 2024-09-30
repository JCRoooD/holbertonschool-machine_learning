#!/usr/bin/env python3
"""This module contains the MultiHeadAttention class"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """This class performs multi-head attention"""
    def __init__(self, dm, h):
        """
        Class constructor
            Args:
                dm (int): the dimensionality of the model
                h (int): the number
        """
        super(MultiHeadAttention, self).__init__()

        # Number of attention heads
        self.h = h

        # Dimensionality of the model
        self.dm = dm

        # Depth of each attention head (dm divided by the number of heads)
        self.depth = dm // h

        # Dense layer to generate the query matrix
        self.Wq = tf.keras.layers.Dense(dm)

        # Dense layer to generate the key matrix
        self.Wk = tf.keras.layers.Dense(dm)

        # Dense layer to generate the value matrix
        self.Wv = tf.keras.layers.Dense(dm)

        # Dense layer to combine the outputs of the attention heads
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """
        Method to call the instance
        """
        # Generate the query matrix using the dense layer
        Q = self.Wq(Q)

        # Generate the key matrix using the dense layer
        K = self.Wk(K)

        # Generate the value matrix using the dense layer
        V = self.Wv(V)

        # Split the query, key, and value matrices into multiple heads
        Q = self.split_heads(Q, self.h)
        K = self.split_heads(K, self.h)
        V = self.split_heads(V, self.h)

        # Apply scaled dot-product attention to each head
        scaled_attention, attention_weights = sdp_attention(Q, K, V, mask)

        # Concatenate the attention outputs from all heads
        scaled_attention = self.combine_heads(scaled_attention)

        # Pass the concatenated output through the final dense layer
        output = self.linear(scaled_attention)

        return output, attention_weights

    def split_heads(self, x, num_heads):
        """
        Splits x into different heads
            Args:
                x (tf.Tensor): contains the input tensor
                num_heads (int): the number of heads
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Reshape x to (batch_size, seq_len, num_heads, depth)
        x = tf.reshape(x, (batch_size, seq_len, num_heads, self.depth))

        # Transpose to (batch_size, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x):
        """
        Inverts the split_heads function
            Args:
                x (tf.Tensor): contains the input tensor
            """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[2]

        # Transpose to (batch_size, seq_len, num_heads, depth)
        x = tf.transpose(x, perm=[0, 2, 1, 3])

        # Reshape to (batch_size, seq_len, dm)
        return tf.reshape(x, (batch_size, seq_len, self.dm))
