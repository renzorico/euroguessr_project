import streamlit as st
import folium
import os
from streamlit_folium import folium_static
from google.cloud import storage

api_key = st.secrets["API_KEY"]
service_account = st.secrets["SERVICE_ACCOUNT"]
bucket_name = st.secrets["BUCKET_NAME"]

def extract_coordinates_from_filename(filename):
    lat_lng = os.path.splitext(filename)[0].split("_")
    latitude = float(lat_lng[0])
    longitude = float(lat_lng[1])
    return latitude, longitude

def display_image_from_gcs(bucket_name, filename):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    image_bytes = blob.download_as_bytes()
    st.image(image_bytes)

def create_map_with_marker(latitude, longitude):
    marker = folium.Map(location=[latitude, longitude], zoom_start=12)
    folium.Marker([latitude, longitude]).add_to(marker)
    return marker

def main():
    # Specify your GCS bucket name
    st.title("Map with Marker")

    # Get the list of image files from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    image_files = [blob.name for blob in bucket.list_blobs()]

    # Select an image from the list
    selected_image = st.selectbox("Select an image", image_files)

    # Extract coordinates from the selected image filename
    latitude, longitude = extract_coordinates_from_filename(selected_image)

    # Display the image
    display_image_from_gcs(bucket_name, selected_image)

    # Create and display the map with the marker
    map_with_marker = create_map_with_marker(latitude, longitude)
    folium_static(map_with_marker)

if __name__ == "__main__":
    main()
