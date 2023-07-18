# import streamlit as st
# from PIL import Image
# from streamlit_folium import folium_static
# from tensorflow.keras.utils import img_to_array
# import folium
# import os
# from google.cloud import storage
# import tensorflow as tf
# import numpy as np


# bucket_name = "world_photos"
# model_path = "world_photos/images"  # Change this to the path of your model files in GCS
# storage_client = storage.Client()
# bucket = storage_client.bucket(bucket_name)

# # Function to load the CNN model from GCS
# def load_cnn_model():
#     # Download the model files from GCS

#     blob = bucket.blob(model_path + "/model.h5")
#     blob.download_to_filename("model.h5")

#     # Load the CNN model using TensorFlow (you can modify this based on your CNN framework)
#     model = tf.keras.models.load_model("model.h5")

#     return model

# # Load and preprocess a new image for prediction
# # from PIL import Image

# def pick_image(filename):
#     blob = bucket.blob(filename)
#     image = blob.download_as_bytes()
#     lat_lng = os.path.splitext(filename)[0].split("_")
#     latitude = float(lat_lng[0])
#     longitude = float(lat_lng[1])
#     return image, latitude, longitude
# # If an other preprocessing was added in the tunning of the model, need to be here as well.
# # Perform the prediction
# # Display the prediction
# #
# # Function to make predictions using the CNN model
# def make_prediction(model, image):
#     # Preprocess the image (resize, normalize, etc.)
#     # Depending on your CNN model and training pipeline, the preprocessing steps may vary.
#     # Here, we'll assume that you have a function called `preprocess_image` for that.

#     # preprocessed_image = preprocess_image(image)
#     # new_image = Image.open('new_image.jpg')  # Replace 'new_image.jpg' with the path to your new image
#     new_image = image.resize((640, 480))  # Resize the image to match the input shape of the model
#     new_image_array = img_to_array(new_image)
#     new_image_array = new_image_array.astype('float32') / 255.0  # Normalize the pixel values between 0 and 1
#     # Make prediction
#     prediction = model.predict(np.expand_dims(new_image_array, axis=0))
#     pred_lat = prediction[0][0]
#     pred_lon = prediction[0][1]
#     return pred_lat, pred_lon

# def extract_coordinates_from_filename(filename):
#     lat_lng = os.path.splitext(filename)[0].split("_")
#     latitude = float(lat_lng[0])
#     longitude = float(lat_lng[1])
#     return latitude, longitude

# def create_map_with_marker(latitude, longitude):
#     # Create a map centered at the given latitude and longitude
#     marker = folium.Map(location=[latitude, longitude], zoom_start=12)
#     # Set the black and white tile layer
#     folium.TileLayer("https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.png", attr="Stamen Toner").add_to(marker)
#     folium.Marker([latitude, longitude]).add_to(marker)
#     return marker

# def main():
#     st.title("Image Upload and Coordinates Extraction")
#     # Upload an image
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg"])

#     if uploaded_file is not None:
#         # Get the filename and extract coordinates
#         filename = uploaded_file.name
#         latitude, longitude = extract_coordinates_from_filename(filename)

#         # Display the uploaded image
#         st.image(uploaded_file, caption=filename, use_column_width=True)

#         st.subheader("Real Coordinates:")
#         st.write("Latitude:", latitude)
#         st.write("Longitude:", longitude)

#         # Create and display the map with the marker
#         map_with_marker = create_map_with_marker(latitude, longitude)
#         st.write("Location on Map:")
#         folium_static(map_with_marker)

# if __name__ == "__main__":
#     main()

#------------------------------------------------------------------------
import streamlit as st
from PIL import Image
from streamlit_folium import folium_static
import folium
import os
from google.cloud import storage
import tensorflow as tf
import numpy as np

bucket_name = "world_photos"
model_path = "Model/500_resnet50_opt5.keras"  # Change this to the path of your model file in GCS
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

# Function to load the CNN model from GCS
def load_cnn_model():
    # Download the model file from GCS224
    blob = bucket.blob(model_path)
    blob.download_to_filename("500_resnet50_opt5.keras")

    # Load the CNN model using TensorFlow (you can modify this based on your CNN framework)
    model = tf.keras.models.load_model("500_resnet50_opt5.keras")

    return model

# Function to preprocess the image (resize, normalize, etc.)
def preprocess_image(image):
    # Perform your custom preprocessing here based on how the model was trained
    # For example, resize the image to the expected input size of your model and normalize the pixel values.
    # Replace the example preprocessing with your actual preprocessing steps.
    preprocessed_image = image.resize((240, 320))
    preprocessed_image = np.array(preprocessed_image)
    preprocessed_image = preprocessed_image.astype('float32') / 255.0  # Normalize the pixel values between 0 and 1
    return preprocessed_image

# Function to make predictions using the CNN model
def make_prediction(model, image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    pred_lat, pred_lon = prediction[0][0], prediction[0][1]
    return pred_lat, pred_lon

# Function to extract coordinates from the image filename
def extract_coordinates_from_filename(filename):
    lat_lng = os.path.splitext(filename)[0].split("_")
    latitude = float(lat_lng[0])
    longitude = float(lat_lng[1])
    return latitude, longitude

# Function to create a map with markers for given and predicted locations
def create_map_with_markers(latitude, longitude, pred_latitude, pred_longitude):
    # Create a map centered at the given latitude and longitude
    marker_map = folium.Map(location=[latitude, longitude], zoom_start=12)
    # Set the black and white tile layer
    # folium.TileLayer("https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.png", attr="Stamen Toner").add_to(marker_map)
    # Add marker for the given location
    folium.Marker([latitude, longitude], popup="Given Location").add_to(marker_map)
    # Add marker for the predicted location
    folium.Marker([pred_latitude, pred_longitude], popup="Predicted Location", icon=folium.Icon(color='red')).add_to(marker_map)
    return marker_map

def main():
    # Load the CNN model
    model = load_cnn_model()

    st.title("Image Upload and Coordinates Extraction")
    # Upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg"])

    if uploaded_file is not None:
        # Get the filename and extract coordinates
        filename = uploaded_file.name
        latitude, longitude = extract_coordinates_from_filename(filename)

        # Display the uploaded image
        st.image(uploaded_file, caption=filename, use_column_width=True)

        # Perform the prediction
        image = Image.open(uploaded_file)
        pred_lat, pred_lon = make_prediction(model, image)

        # Display the given and predicted latitude and longitude
        st.subheader("Given Coordinates:")
        st.write("Latitude:", latitude)
        st.write("Longitude:", longitude)
        st.subheader("Predicted Coordinates:")
        st.write("Predicted Latitude:", pred_lat)
        st.write("Predicted Longitude:", pred_lon)

        # Create and display the map with markers for the given and predicted locations
        map_with_markers = create_map_with_markers(latitude, longitude, pred_lat, pred_lon)
        st.write("Location on Map:")
        folium_static(map_with_markers)

if __name__ == "__main__":
    main()

#------------------------------------------------------------------------
# import streamlit as st
# from PIL import Image
# from streamlit_folium import folium_static
# import folium
# import os
# from google.cloud import storage
# import tensorflow as tf
# import numpy as np

# bucket_name = "world_photos"
# model_path = "Model/500_resnet50_opt5.keras"  # Change this to the path of your model file in GCS
# storage_client = storage.Client()
# bucket = storage_client.bucket(bucket_name)

# # Function to load the CNN model from GCS
# def load_cnn_model():
#     # Download the model file from GCS224
#     blob = bucket.blob(model_path)
#     blob.download_to_filename("500_resnet50_opt5.keras")

#     # Load the CNN model using TensorFlow (you can modify this based on your CNN framework)
#     model = tf.keras.models.load_model("500_resnet50_opt5.keras")

#     return model

# # Function to preprocess the image (resize, normalize, etc.)
# def preprocess_image(image):
#     # Perform your custom preprocessing here based on how the model was trained
#     # For example, resize the image to the expected input size of your model and normalize the pixel values.
#     # Replace the example preprocessing with your actual preprocessing steps.
#     preprocessed_image = image.resize((240, 320))
#     preprocessed_image = np.array(preprocessed_image)
#     preprocessed_image = preprocessed_image.astype('float32') / 255.0  # Normalize the pixel values between 0 and 1
#     return preprocessed_image

# # Function to make predictions using the CNN model
# def make_prediction(model, image):
#     preprocessed_image = preprocess_image(image)
#     prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
#     pred_lat, pred_lon = prediction[0][0], prediction[0][1]
#     return pred_lat, pred_lon

# # Function to extract coordinates from the image filename
# def extract_coordinates_from_filename(filename):
#     lat_lng = os.path.splitext(filename)[0].split("_")
#     latitude = float(lat_lng[0])
#     longitude = float(lat_lng[1])
#     return latitude, longitude

# # Function to create a map with markers for given and predicted locations
# def create_map_with_markers(latitude, longitude, pred_latitude, pred_longitude):
#     # Create a map centered at the given latitude and longitude
#     marker_map = folium.Map(location=[latitude, longitude], zoom_start=12)
#     # Set the black and white tile layer
#     # folium.TileLayer("https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.png", attr="Stamen Toner").add_to(marker_map)
#     # Add marker for the given location
#     folium.Marker([latitude, longitude], popup="Given Location").add_to(marker_map)
#     # Add marker for the predicted location
#     folium.Marker([pred_latitude, pred_longitude], popup="Predicted Location", icon=folium.Icon(color='red')).add_to(marker_map)
#     return marker_map

# def main():
#     # Load the CNN model
#     model = load_cnn_model()

#     st.title("Image Upload and Coordinates Extraction")
#     # Upload an image
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg"])

#     # Sidebar input for manual latitude and longitude
#     st.sidebar.subheader("Manually Input Latitude and Longitude")
#     manual_latitude = st.sidebar.number_input("Latitude", value=0.0)
#     manual_longitude = st.sidebar.number_input("Longitude", value=0.0)

#     if uploaded_file is not None:
#         # Get the filename and extract coordinates
#         filename = uploaded_file.name
#         latitude, longitude = extract_coordinates_from_filename(filename)

#         # Display the uploaded image
#         st.image(uploaded_file, caption=filename, use_column_width=True)

#         # Perform the prediction
#         image = Image.open(uploaded_file)
#         pred_lat, pred_lon = make_prediction(model, image)

#         # Display the given and predicted latitude and longitude
#         st.subheader("Given Coordinates:")
#         st.write("Latitude:", latitude)
#         st.write("Longitude:", longitude)
#         st.subheader("Predicted Coordinates:")
#         st.write("Predicted Latitude:", pred_lat)
#         st.write("Predicted Longitude:", pred_lon)

#         # Create and display the map with markers for the given and predicted locations
#         if manual_latitude != 0.0 and manual_longitude != 0.0:
#             # Use manually entered latitude and longitude if provided
#             latitude = manual_latitude
#             longitude = manual_longitude

#         map_with_markers = create_map_with_markers(latitude, longitude, pred_lat, pred_lon)
#         st.write("Location on Map:")
#         folium_static(map_with_markers)

# if __name__ == "__main__":
#     main()
