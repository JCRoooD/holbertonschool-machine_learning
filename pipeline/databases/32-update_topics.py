#!/usr/bin/env python3
""" Update topics module"""


def update_topics(mongo_collection, name, topics):
    """
    Function that updates a document in a collection based on kwargs
    Args:
        mongo_collection: pymongo collection object
        name: string with the name to search
        topics: list of strings to change
    Returns:
    Nothing
    """
    mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}}
    )
