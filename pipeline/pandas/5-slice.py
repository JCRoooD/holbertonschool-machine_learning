#!/usr/bin/env python3
""" slice module to slice a pd.DataFrame"""


def slice_data(df):
    """ function that slices a pd.DataFrame"""
    return df[[ 'High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
