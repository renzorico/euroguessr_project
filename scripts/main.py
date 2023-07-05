import requests
import os
from coordinates import read_coordinates_from_file
from api import api_call
from google.cloud import storage
from dotenv import load_dotenv

# Set your API key
load_dotenv('.env')
api_key = os.getenv('API_KEY')
service_account = os.getenv('SERVICE_ACCOUNT')
file_path = "../raw_data"

# Set the location parameters
coordinates = read_coordinates_from_file(file_path)

for i in range(20):
    coord = coordinates[1]
    latitude, longitude = coord
    api_call(latitude, longitude, api_key, service_account)
    

