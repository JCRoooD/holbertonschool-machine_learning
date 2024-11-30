#!/usr/bin/env python3
""" array module"""


def array(df):
    """ function that creates an array from a pd.DataFrame"""
    return df[['High', 'Close']].tail(10).to_numpy()
