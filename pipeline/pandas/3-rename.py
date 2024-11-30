#!/usr/bin/env python3
""" Rename module"""
import pandas as pd


def rename_columns(df, columns):
    """ function renames columns of a pd.DataFrame"""
    # rename columns
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # datetime column to datetime objects
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # display 
    df = df[['Datetime', 'Close']]

    return df
