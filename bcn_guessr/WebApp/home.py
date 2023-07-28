import streamlit as st
from geoLSTM import Geoguessr
from PIL import Image
import pickle
import os
import tempfile
import folium
from streamlit_folium import folium_static

# Function to load the model
def load_model(model_path):
    return Geoguessr.load(model_path)

# Prediction function for Model 1
def predict_model_1(uploaded_files):
    # Load the model
    model_path = "models/model_2.473_37.h5"  # Replace with the path to your model
    geo_model = load_model(model_path)
    # Create a temporary directory to store the uploaded images
    temp_dir = tempfile.TemporaryDirectory()
    # Save the uploaded images to the temporary directory and get the file paths
    file_paths = []
    for idx, file in enumerate(uploaded_files):
        file_path = os.path.join(temp_dir.name, f"uploaded_image_{idx}.jpg")
        with open(file_path, "wb") as f:
            f.write(file.read())
        file_paths.append(file_path) 
    # Display the uploaded images on a small size
    st.write("Uploaded Images:")
    image_col1, image_col2, image_col3 = st.columns(3)
    for idx, file_path in enumerate(file_paths):
        image = Image.open(file_path)
        if idx == 0:
            image_col1.image(image, caption=f"Image {idx+1}", width=200)
        elif idx == 1:
            image_col2.image(image, caption=f"Image {idx+1}", width=200)
        elif idx == 2:
            image_col3.image(image, caption=f"Image {idx+1}", width=200)

    BcnPolyGrid = pickle.load(open("/Users/Corcho/code/Bcnguessr/data/BcnPolyGrid.pkl", 'rb'))  # Replace with the path to your BcnPolyGrid.pkl file
    # Perform the prediction and get the plot path
    location, pred_plot = geo_model.GridPrediction(file_paths, BcnPolyGrid)
    
    # Remove the temporary directory and its contents
    temp_dir.cleanup()
    
    return location, pred_plot

# Prediction function for Model 2
def predict_model_2(uploaded_files):
    # Load the model
    model_path = "models/model_2.77_26.h5"  # Replace with the path to your model
    geo_model = load_model(model_path)
    # Create a temporary directory to store the uploaded images
    temp_dir = tempfile.TemporaryDirectory()
    # Save the uploaded images to the temporary directory and get the file paths
    file_paths = []
    for idx, file in enumerate(uploaded_files):
        file_path = os.path.join(temp_dir.name, f"uploaded_image_{idx}.jpg")
        with open(file_path, "wb") as f:
            f.write(file.read())
        file_paths.append(file_path) 
    # Display the uploaded images on a small size
    st.write("Uploaded Images:")
    image_col1, image_col2, image_col3 = st.columns(3)
    for idx, file_path in enumerate(file_paths):
        image = Image.open(file_path)
        if idx == 0:
            image_col1.image(image, caption=f"Image {idx+1}", width=200)
        elif idx == 1:
            image_col2.image(image, caption=f"Image {idx+1}", width=200)
        elif idx == 2:
            image_col3.image(image, caption=f"Image {idx+1}", width=200)

    BcnPolyGrid = pickle.load(open("/Users/Corcho/code/Bcnguessr/data/BcnPolyGrid.pkl", 'rb'))  # Replace with the path to your BcnPolyGrid.pkl file
    # Perform the prediction and get the plot path
    location, pred_plot = geo_model.GridPrediction(file_paths, BcnPolyGrid)
    
    # Remove the temporary directory and its contents
    temp_dir.cleanup()
    
    return location, pred_plot

# Prediction function for Model 3
def predict_model_3(uploaded_files):
    # Load the model
    model_path = "models/model_2.628_24.h5"  # Replace with the path to your model
    geo_model = load_model(model_path)
    # Create a temporary directory to store the uploaded images
    temp_dir = tempfile.TemporaryDirectory()
    # Save the uploaded images to the temporary directory and get the file paths
    file_paths = []
    for idx, file in enumerate(uploaded_files):
        file_path = os.path.join(temp_dir.name, f"uploaded_image_{idx}.jpg")
        with open(file_path, "wb") as f:
            f.write(file.read())
        file_paths.append(file_path) 
    # Display the uploaded images on a small size
    st.write("Uploaded Images:")
    image_col1, image_col2, image_col3 = st.columns(3)
    for idx, file_path in enumerate(file_paths):
        image = Image.open(file_path)
        if idx == 0:
            image_col1.image(image, caption=f"Image {idx+1}", width=200)
        elif idx == 1:
            image_col2.image(image, caption=f"Image {idx+1}", width=200)
        elif idx == 2:
            image_col3.image(image, caption=f"Image {idx+1}", width=200)

    BcnPolyGrid = pickle.load(open("/Users/Corcho/code/Bcnguessr/data/BcnPolyGrid.pkl", 'rb'))  # Replace with the path to your BcnPolyGrid.pkl file
    # Perform the prediction and get the plot path
    location, pred_plot = geo_model.GridPrediction(file_paths, BcnPolyGrid)
    
    # Remove the temporary directory and its contents
    temp_dir.cleanup()
    
    return location, pred_plot

# Prediction function for Model 4
def predict_model_4(uploaded_files):
    # Load the model
    model_path = "models/model_3.972_37.h5"  # Replace with the path to your model
    geo_model = load_model(model_path)
    # Create a temporary directory to store the uploaded images
    temp_dir = tempfile.TemporaryDirectory()
    # Save the uploaded images to the temporary directory and get the file paths
    file_paths = []
    for idx, file in enumerate(uploaded_files):
        file_path = os.path.join(temp_dir.name, f"uploaded_image_{idx}.jpg")
        with open(file_path, "wb") as f:
            f.write(file.read())
        file_paths.append(file_path) 
    # Display the uploaded images on a small size
    st.write("Uploaded Images:")
    image_col1, image_col2, image_col3 = st.columns(3)
    for idx, file_path in enumerate(file_paths):
        image = Image.open(file_path)
        if idx == 0:
            image_col1.image(image, caption=f"Image {idx+1}", width=200)
        elif idx == 1:
            image_col2.image(image, caption=f"Image {idx+1}", width=200)
        elif idx == 2:
            image_col3.image(image, caption=f"Image {idx+1}", width=200)

    BcnPolyGrid = pickle.load(open("/Users/Corcho/code/Bcnguessr/data/BcnPolyGrid.pkl", 'rb'))  # Replace with the path to your BcnPolyGrid.pkl file
    # Perform the prediction and get the plot path
    location, pred_plot = geo_model.GridPrediction(file_paths, BcnPolyGrid)
    
    # Remove the temporary directory and its contents
    temp_dir.cleanup()
    
    return location, pred_plot


# Main Streamlit app
def main():
    # Add a title to the app
    st.title("Barcelona GeoGuessr Model Prediction")
    
    # Ask the user to choose the model for prediction
    model_choice = st.radio("Select a Model:", ("Model 1", "Model 2", "Model 3", "Model 4"))

    # Ask the user to upload the triplet image
    st.write("Upload the triplet image:")
    uploaded_files = st.file_uploader("Choose a triplet image file...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    # Make predictions on the uploaded images and display the result
    if uploaded_files is not None and len(uploaded_files) == 3:
        if model_choice == "Model 1":
            location, pred_plot = predict_model_1(uploaded_files)
        elif model_choice == "Model 2":
            location, pred_plot = predict_model_2(uploaded_files)
        elif model_choice == "Model 3":
            location, pred_plot = predict_model_3(uploaded_files)
        elif model_choice == "Model 4":
            location, pred_plot = predict_model_4(uploaded_files)

        ## Second Secction ##
        
        st.write("Predicted location: ", location)
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(pred_plot)

        st.write("Prediction Confirmed!")
        
        latitude, longitude = map(float, location.split(','))
        
        # Create a map centered at the user-provided latitude and longitude
        my_map = folium.Map(location=[latitude, longitude], zoom_start=13)
        
        # Add a marker at the specified location
        red_marker = folium.Icon(color='red', icon='info-sign')
        folium.Marker([latitude, longitude], popup='Selected Location', icon=red_marker).add_to(my_map)
        
        # Display the map in Streamlit
        folium_static(my_map, width=725)

        

if __name__ == '__main__':
    main()

