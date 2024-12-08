#!/usr/bin/env python3
""" Sentience module"""
import requests


def sentientPlanets():
    """ 
    Function that returns a list of planets that have sentient life
    Args:
        None
    Returns:
    List of planets that have sentient life
    """
    url = "https://swapi.dev/api/planets/"
    # Initialize an empty list to store the names of planets that have sentient life
    planets = []

    # Loop to paginate through all pages of the API response
    while url:
        # Send a GET request to the current URL
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code != 200:
            break

        # Parse the JSON response
        data = response.json()
        # Iterate through the list of planets in the current page of the response
        for planet in data.get('results', []):
            try:
                if planet.get('population').isdigit() and int(planet.get('population')) > 1000000000:
                    planets.append(planet.get('name'))
            except ValueError:
                continue

        # Get the URL for the next page of results
        url = data.get('next')

    # Return the list of planets that have sentient life
    return planets
