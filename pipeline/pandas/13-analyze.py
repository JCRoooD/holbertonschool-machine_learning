#!/usr/bin/env python3
""" analyze module"""


def analyze(df):
    """ function that analyzes the pd.DataFrame"""
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    return df.describe()
