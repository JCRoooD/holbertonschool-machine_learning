#!/bin/usr/env python3
""" concat module"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """ function that concatenates two pd.DataFrames"""
    
    df1 = index(df1)
    df2 = index(df2)

    # Concatenate the two pd.DataFrames
    df2 = df2.loc[:1417411920]
    return pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
