#!/bin/usr/env python3
""" prune module"""


def prune(df):
    """ function that prunes the pd.DataFrame"""
    return df.dropna(subset=['Close'])
