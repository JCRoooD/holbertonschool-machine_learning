#!/usr/bin/env python3
""" User location module"""
import requests
import sys
from datetime import datetime\


def user_location(ip):
    """ 
    Function that returns the location of a user based on their IP address
    Args:
        ip: IP address of the user
    Returns:
    Location of the user
    """
    response = requests.get(api_url)
    # Check if the request was successful
    if response.status_code == 404:
        print("Not found")

    # If rate limit is exceeded (403)
    elif response.status_code == 403:
        reset_time = int(response.headers.get('X-RateLimit-Reset'))
        current_time = int(datetime.utcnow().timestamp())
        reset_in_minutes = (reset_time - current_time) // 60
        print(f"Reset in {reset_in_minutes} min")

    # If the user is found (200)
    elif response.status_code == 200:
        user_data = response.json()
        location = user_data.get("location")

        if location:
            print(location)
        else:
            print("No location found")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API URL>")
        sys.exit(1)

    api_url = sys.argv[1]
    get_user_location(api_url)
