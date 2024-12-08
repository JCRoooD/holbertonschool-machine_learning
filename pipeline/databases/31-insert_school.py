#!/usr/bin/env python3
""" Insert school module"""


def insert_school(mongo_collection, **kwargs):
    """
    Function that inserts a new document in a collection based on kwargs
    Args:
        mongo_collection: pymongo collection object
        kwargs: dictionary of arguments to insert
    Returns:
    The new _id
    """
    return mongo_collection.insert_one(kwargs).inserted_id
