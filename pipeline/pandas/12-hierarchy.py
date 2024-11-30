#!/usr/bin/env python3
""" hierarchy module"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """ function that sets the index of the pd.DataFrame"""
    df1 = index(df1)
    df2 = index(df2)

    # Concatenate the two pd.DataFrames
    df1 = df1.loc[:1417411980:1417417980]
    df2 = df2.loc[:1417411980:1417417980]

    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    df = df.reorder_levels([1, 0], axis=0)

    return df.sort_index()
