#!/usr/bin/env python3
""" 33-schools_by_topic 
    Function that returns the list of school having a specific topic
"""


def schools_by_topic(mongo_collection, topic):
    """
    Function that returns the list of school having a specific topic
    Args:
        mongo_collection: pymongo collection object
        topic: string topic to search
    Returns:
    List of schools
    """
    return list(mongo_collection.find({"topics": topic}))
