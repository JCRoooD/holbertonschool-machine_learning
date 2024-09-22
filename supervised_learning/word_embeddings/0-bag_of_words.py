#!/usr/bin/env python3
""" 0. Bag Of Words module"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """ Function that creates a bag of words embedding matrix"""
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    word_embedd = X.toarray()
    feature_name = vectorizer.get_feature_names()
    return word_embedd, feature_name

