#!/usr/bin/env python3
"""This modle contain the cumulative_bleu function"""
import numpy as np
from collections import Counter


def ngrams(sequence, n):
    """Generate n-grams from a sequence of words."""
    return [tuple(sequence[i: i + n]) for i in range(len(sequence) - n + 1)]


def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence."""
    # Generate n-grams for the sentence
    sentence_ngrams = ngrams(sentence, n)
    sentence_ngram_counts = Counter(sentence_ngrams)

    # Generate n-grams for the references
    max_ngram_counts = Counter()
    for reference in references:
        reference_ngrams = ngrams(reference, n)
        reference_ngram_counts = Counter(reference_ngrams)
        for ngram in reference_ngram_counts:
            max_ngram_counts[ngram] = max(
                max_ngram_counts[ngram], reference_ngram_counts[ngram]
            )

    # Calculate clipped counts
    clipped_ngram_counts = {
        ngram: min(count, max_ngram_counts[ngram])
        for ngram, count in sentence_ngram_counts.items()
    }

    # Calculate precision
    precision = sum(
        clipped_ngram_counts.values()) / max(1, len(sentence_ngrams))

    return precision


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.
    Returns:
        float: The cumulative n-gram BLEU score.
    """
    precisions = []
    for i in range(1, n + 1):
        precisions.append(ngram_bleu(references, sentence, i))

    # Calculate geometric mean of precisions
    geometric_mean = np.exp(np.sum((1 / n) * np.log(precisions)))

    # Calculate brevity penalty
    ref_lengths = [len(ref) for ref in references]
    closest_ref_length = min(
        ref_lengths, key=lambda ref_len: (
            abs(ref_len - len(sentence)), ref_len)
    )
    if len(sentence) > closest_ref_length:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_length / len(sentence))

    # Calculate cumulative BLEU score
    bleu_score = brevity_penalty * geometric_mean

    return bleu_score
