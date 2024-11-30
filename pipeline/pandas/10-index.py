#!/usr/bin/env python3
""" index module"""


def index(df):
    """ function that sets the index of the pd.DataFrame"""
    if 'Timestamp' in df.columns:
        df = df.set_index('Timestamp')
    return df
