from coordinates import generate_random_coordinates
from google.cloud import storage
import requests

# Set your API key
api_key = "YOUR_API_KEY"

# Set the location parameters
latitude, longitude = generate_random_coordinates()

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

# Save the image to a file
with open("street_view_image.jpg", "wb") as f:
    f.write(response.content)

# Set up the Google Cloud Storage client

storage_client = storage.Client.from_service_account_json('path/to/service_account_key.json')

# Replace with your actual bucket name
bucket_name = 'your-bucket-name'

# Replace with the path where you have saved the image
image_path = 'path/to/image.jpg'

bucket = storage_client.get_bucket(bucket_name)

# Provide the desired path within the bucket for the image
blob = bucket.blob('path/in/bucket/image.jpg')
blob.upload_from_filename(image_path)
