#!/usr/bin/env python3
"""This module creates a word2vec model"""
import gensim


def word2vec_model(
    sentences,
    vector_size=100,
    min_count=5,
    window=5,
    negative=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1,
):
    """This function creates builds and trains a word2vec model
    """
    # initialize the Word2Vec model using gensim
    model = gensim.models.Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=0 if cbow else 1,
        seed=seed,
        workers=workers,
        epochs=epochs,
    )

    model.build_vocab(sentences)
    # train the model
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
