import requests
import os
from google.cloud import storage
from dotenv import load_dotenv

# Third step: Called GSV API to bring the correspondant images for every coordinate, storaged the images in a 'lat_lon.jpeg' format name in a 'coordinates_done.txt' file.
# In case there is an error downloading the images, when the process starts again, it will check if the coordinates of the image already exists in the coordinates_done.txt file so it doesn't bring the same image twice.



# Set your API key
load_dotenv('.env')
api_key = os.getenv('API_KEY')
service_account = os.getenv('SERVICE_ACCOUNT')

def api_call(latitude, longitude, api_key, service_account):
    # Set the desired image size
    image_height = 480
    image_width = 640

    # Set the heading, pitch, and fov parameters
    heading = 90
    pitch = 0
    fov = 90

    # Construct the URL for the API request
    url = f"https://maps.googleapis.com/maps/api/streetview?size={image_width}x{image_height}&location={latitude},{longitude}&heading={heading}&pitch={pitch}&fov={fov}&key={api_key}"

    # Convert latitude and longitude to strings
    lat_str = str(latitude)
    lon_str = str(longitude)

    # Concatenate latitude, longitude, and file extension
    filename = f"{lat_str}_{lon_str}.jpg"
    # Send the request and receive the response
    response = requests.get(url)

    # Set up the Google Cloud Storage client
    storage_client = storage.Client.from_service_account_json(service_account)

    # Replace with your actual bucket name
    bucket_name = os.getenv('BUCKET_NAME')

    # Check if the response is successful (status code 200)
    if response.status_code == 200:
        # Provide the desired path within the bucket for the images
        image_path = f'images/{filename}'
        bucket = storage_client.bucket(bucket_name)
        # Upload the image directly to the bucket
        blob = bucket.blob(image_path)
        blob.upload_from_string(response.content, content_type='image/jpeg')
        print("Image saved succesfully in Google Storage")
    else:
        print ("Error: Unable to fetch the street view image.")
