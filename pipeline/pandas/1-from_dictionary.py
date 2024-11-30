#!/usr/bin/env python3
""" From Dictionary module"""
import pandas as pd


# Create a pd.DataFrame from a dictionary
dict = {    'First': [0.0, 0.5, 1.0, 1.5],
            'Second': ['one', 'two', 'three', 'four']}

index = ['A', 'B', 'C', 'D']

df = pd.DataFrame(dict, index=index)
