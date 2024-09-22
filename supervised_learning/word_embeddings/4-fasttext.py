#!/usr/bin/env python3
"""This module creates a fasttext model"""
import gensim


def fasttext_model(
    sentences,
    vector_size=100,
    min_count=5,
    negative=5,
    window=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1,
):
    """function that creates and trains a gensim fasttext model

    Args:
        sentences (list): list of sentences to be trained on
        vector_size (int): dimension of the word vectors
        min_count (int): minimum number of occurrences of a word
        for it to be included in the model
        negative (int): number of negative samples
        window (int): maximum distance between the current
        and predicted word within a sentence
        cbow (bool): determines the training algorithm. If True,
        CBOW is used; if False, skip-gram is used
        epochs (int): number of iterations over the corpus
        seed (int): seed for the random number generator
        workers (int): number of worker threads to train the model
    """
    # create the fasttext model
    model = gensim.models.FastText(
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
    # build the vocabulary
    model.build_vocab(sentences)
    # train the model
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
