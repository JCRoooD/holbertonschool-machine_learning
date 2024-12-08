#!/usr/bin/env python3
""" Passengers module"""
import requests


def availableShips(passengerCount):
    """ availableShips function
    Args:
        passengerCount: Number of passengers
    Returns:
    List of ships that can carry the given number of passengers
    """
    url = "https://swapi.dev/api/starships/"
    # Initialize an empty list to store the names of ships that can
    # carry the given number of passengers
    ships = []

    # Loop to paginate through all pages of the API response
    while url:
        # Send a GET request to the current URL
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code != 200:
            break

        # Parse the JSON response
        data = response.json()
        # Iterate through the list of ships in the current page of the response
        for ship in data.get('results', []):
            try:

                passengers = ship.get('passengers', '0').replace(',', '')

                if passengers.isdigit() and int(passengers) >= passengerCount:
                    ships.append(ship.get('name'))
            except ValueError:

                continue

        # Get the URL for the next page of results
        url = data.get('next')

    # Return the list of ships that can carry the given number of passengers
    return ships
