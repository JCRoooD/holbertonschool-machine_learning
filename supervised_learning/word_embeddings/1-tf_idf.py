#!/usr/bin/env python3
""" this module contains the function tf_idf """
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    creates a TF-IDF embedding
    :param sentences: list of sentences to analyze
    :param vocab: list of the vocabulary words to use for the analysis
    :return: embeddings, features
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    word_embedd = X.toarray()
    feature_name = vectorizer.get_feature_names_out()
    return word_embedd, feature_name
