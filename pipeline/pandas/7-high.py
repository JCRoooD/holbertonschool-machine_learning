#!/usr/bin/env python3
""" high module, set High column to 0"""


def high(df):
    """ function sorts a dataframe by the High column"""
    return df.sort_values(by='High', ascending=False)
