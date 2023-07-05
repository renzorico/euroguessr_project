import requests
import os
from coordinates import generate_random_coordinates
from google.cloud import storage
from dotenv import load_dotenv

# Set your API key
load_dotenv('.env')
api_key = os.getenv('API_KEY')
service_account = os.getenv('SERVICE_ACCOUNT')

# Set the location parameters
latitude, longitude = 40.300000000000075,-3.700000000000021

# Set the desired image size
image_width = 640
image_height = 480

# Set the heading, pitch, and fov parameters
heading = 90
pitch = 0
fov = 90

# Construct the URL for the API request
url = f"https://maps.googleapis.com/maps/api/streetview?size={image_width}x{image_height}&location={latitude},{longitude}&heading={heading}&pitch={pitch}&fov={fov}&key={api_key}"

# Send the request and receive the response
response = requests.get(url)

# Set up the Google Cloud Storage client

storage_client = storage.Client.from_service_account_json(service_account)

# Replace with your actual bucket name
bucket_name = os.getenv('BUCKET_NAME')
# Provide the desired path within the bucket for the image
image_path = 'images/image1.jpg'

bucket = storage_client.bucket(bucket_name)

# Upload the image directly to the bucket
blob = bucket.blob(image_path)
blob.upload_from_string(response.content, content_type='image/jpeg')
