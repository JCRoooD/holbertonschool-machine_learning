#!/usr/bin/env python3
""" stast abput Nginx logs """
from pymongo import MongoClient


def log_stats(logs_collection):
    """ for collection logs 
    Args:
        logs_collection: pymongo collection object
        Returns:
        Nothing
    """
    total_logs = logs_collection.count_documents({})
    print(f"{total_logs} logs")

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods:")
    for method in methods:
        count = logs_collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")

    status_check_count = logs_collection.count_documents(
        {"method": "GET", "path": "/status"}
    )
    print(f"{status_check_count} status check")


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    db = client.logs
    logs_collection = db.nginx
    log_stats(logs_collection)
