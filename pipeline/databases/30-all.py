#!/usr/bin/env python3
""" All databases module"""


def list_all(mongo_collection):
    """
    Function that lists all documents in a collection
    Args:
        mongo_collection: pymongo collection object
    Returns:
    List of documents in the collection
    """
    return list(mongo_collection.find())
