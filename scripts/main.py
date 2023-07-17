import os
from coordinates_reader import read_coordinates_from_file
from api import api_call
from dotenv import load_dotenv

# Set your API key
load_dotenv('../.env')
api_key = os.getenv('API_KEY')
service_account = os.getenv('SERVICE_ACCOUNT')
file_path = "barcelona_coordinates2.txt"
coordinates_done_file = "coordinates_done.txt"

# Set the location parameters
coordinates = read_coordinates_from_file(file_path)

# Load the coordinates_done from file if it exists
if os.path.isfile(coordinates_done_file):
    with open(coordinates_done_file, 'r') as f:
        coordinates_done = [tuple(map(float, line.strip().split(','))) for line in f]
else:
    coordinates_done = []

for i, coord in enumerate(coordinates[:]):
    if coord in coordinates_done:
        print('Already in GCS')
    else:
        latitude, longitude = coord
        print(i)
        api_call(latitude, longitude, api_key, service_account)
        coordinates_done.append(coord)

# Save the updated coordinates_done list to file
with open(coordinates_done_file, 'w') as f:
    for coord in coordinates_done:
        f.write(f'{coord[0]},{coord[1]}\n')
