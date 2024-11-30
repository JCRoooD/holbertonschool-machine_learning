#!/bin/usr/env python3
""" fill module"""


def fill(df):
    """ this function fills missing values in a pd.DataFrame"""
    if "Weighted_Price" in df.columns:
        df = df.drop(columns=["Weighted_Price"])

    # Fill missing values in the "Close" column with the previous row's value
    if "Close" in df.columns:
        df["Close"] = df["Close"].ffill()

    # Fill missing values in these columns with corresponding "Close" values
    for column in ["High", "Low", "Open"]:
        if column in df.columns:
            df[column] = df[column].fillna(df["Close"])

    # Set missing values in the Volume columns to 0
    for column in ["Volume_(BTC)", "Volume_(Currency)"]:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    return df
