import random
import pickle
from google.cloud import storage
from tensorflow.keras.utils import img_to_array
from PIL import Image
import io
import glob

def get_data_from_GCS(n_images, image_size=(240, 320)):
    # Initialize Google Cloud Storage client
    client = storage.Client()

    # Specify the bucket name where your images are located
    bucket_name = 'world_photos'

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # Create an empty dictionary to store the image data
    euro_dict = {'images': [], 'target': []}

    # Specify the subfolder path where your images are located
    subfolder = 'images/'

    # List all blobs (files) in the bucket with the specified prefix (subfolder)
    blobs = bucket.list_blobs(prefix=subfolder)

    # Get a list of all blob names
    blob_names = [blob.name for blob in blobs if not blob.name.endswith('/')]

    # Randomly select n_images from the blob names without repetition
    selected_blob_names = random.sample(blob_names, n_images)

    # Iterate through each selected blob name
    for blob_name in selected_blob_names:
        # Download the image data as bytes
        blob = bucket.blob(blob_name)
        image_data = blob.download_as_bytes()

        # Open the image using PIL
        image = Image.open(io.BytesIO(image_data))
        
        # Resize the image to the desired size
        image = image.resize(image_size)

        # Convert the image to a numpy array
        img_array = img_to_array(image)

        # Append the image array to the 'images' list in the dictionary
        euro_dict['images'].append(img_array)

        # Extract the target name from the blob name (assuming it follows the same format)
        target = blob_name.replace('.jpg', '').split('/')[-1]

        # Append the target name to the 'target' list in the dictionary
        euro_dict['target'].append(target)

    # Save the dictionary as a pickle file
    with open(f'/mnt/mydisk/{n_images}_euro_dict.pkl', 'wb') as file:    
        pickle.dump(euro_dict, file)

#Each time you run this code will create a new .pkl file with 100 random images from our GCS
#get_data_from_GCS(500)

def get_sample_dict():
    # Find the first pickle file in the current directory
    pickle_files = glob.glob("/mnt/mydisk/*.pkl")
    if len(pickle_files) > 0:
        pickle_file = pickle_files[0]
        # Open the pickle file in read mode
        with open(pickle_file, 'rb') as file:
            # Load the dictionary from the pickle file
            euro_dict = pickle.load(file)
        return euro_dict
    else:
        print("No pickle files found.")
        return None
    
