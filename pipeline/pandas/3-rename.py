#!/usr/bin/env python3
""" Rename module"""
import pandas as pd


def rename_columns(df, columns):
    """ function renames columns of a pd.DataFrame"""
    # rename columns
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # Convert the 'Datetime' column to datetime objects
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # Display only the 'Datetime' and 'Close' columns
    df = df[['Datetime', 'Close']]

    return df
