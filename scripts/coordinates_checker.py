import requests
from dotenv import load_dotenv
import os
import requests

load_dotenv('.env')

# First step: Check all the coordinates that have images available in Google Street View API in the specific polygon ((lat_start, lat_end),(lng_start, lng_end)).
# Second step: Saved those (OK) sets of coordinates in a file called 'barcelona_coordinates2.txt'.

def save_coordinates_to_file(coordinates, filename):
    with open(filename, 'w') as file:
        for lat, lng in coordinates:
            file.write(f'{lat},{lng}\n')

def get_available_coordinates(region):
    coordinates_list = []

    # Define the latitude and longitude boundaries for the region
    lat_start, lat_end = 41.35, 41.45
    lng_start, lng_end = 2.10, 2.25

    # Define the precision of the coordinates
    precision = 3

    # Define the step size for generating the grid
    step_size = 10**(-precision)

    # Generate the grid of coordinates
    lat = lat_start
    while lat <= lat_end:
        lng = lng_start
        while lng <= lng_end:
            # Construct the coordinate with the desired precision
            coordinate = round(lat, precision), round(lng, precision)

            # Make a request to the Street View Image Metadata API
            api_key = os.getenv('API_KEY')  # Replace with your own API key
            url = f'https://maps.googleapis.com/maps/api/streetview/metadata?location={coordinate[0]},{coordinate[1]}&key={api_key}'
            response = requests.get(url)
            metadata = response.json()

            # Check if a valid image is available (status code OK)
            if metadata.get('status') == 'OK':
                coordinates_list.append(coordinate)

                # Check if the desired number of coordinates is reached
                if len(coordinates_list) >= 10000:
                    return coordinates_list

            lng += step_size
        lat += step_size

    return coordinates_list

# Example usage
barcelona_coordinates = get_available_coordinates('Barcelona')

# Print the number of coordinates found
print(f'Number of available coordinates: {len(barcelona_coordinates)}')

save_coordinates_to_file(barcelona_coordinates, 'barcelona_coordinates2.txt')
