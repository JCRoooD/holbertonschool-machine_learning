#!/usr/bin/env python3
""" 2. Train Word2Vec """
from gensim


def word2vec_model(sentences,
                size=100,
                min_count=5,
                window=5,
                negative=5,
                cbow=True,
                iterations=5,
                seed=0,
                workers=1):
    """function that creates and trains a gensim word2vec model"""
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
    # build the vocabulary
    # this hepls to create the one-hot encoding of the words
    model.build_vocab(sentences)
    # train the model
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
