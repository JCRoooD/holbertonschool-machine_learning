#!/usr/bin/env python3
""" 3. Gensim to keras """
import tensorflow as tf


def gensim_to_keras(model):
    """function that converts a gensim word2vec
    model to a keras Embedding layer"""
    weights = model.wv.vectors

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )
    return embedding_layer
