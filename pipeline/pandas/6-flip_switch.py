#!/usr/bin/env python3
""" flip_switch module"""


def flip_switch(df):
    """ function that switches the High and Low columns"""
    return df.sort_index(ascending=False).transpose()
