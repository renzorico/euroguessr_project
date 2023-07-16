# First Model, Using a Sequential basic CNN model

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from utils import get_sample_dict
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import mean_squared_error

euro_dict = get_sample_dict()

# Convert the list of images into a NumPy array
images = np.array(euro_dict['images'])

# Normalize the pixel values between 0 and 1
images = images.astype('float32') / 255.0

# Extract the target values
targets = euro_dict['target']

# Split the target values and convert latitude and longitude to float
targets = [target.split('_') for target in targets]
targets = [[float(coord) for coord in target] for target in targets]

# Convert the target values to a NumPy array
targets = np.array(targets)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=42)

model = Sequential()
# Add a convolutional layer with 32 filters, a 3x3 kernel, and 'relu' activation
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(480, 640, 3)))
# Add a max pooling layer with a 2x2 pool size
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten the previous layer output
model.add(Flatten())
# Add a fully connected layer with 64 units and 'relu' activation
model.add(Dense(64, activation='relu'))
# Add a fully connected layer with 128 units and 'relu' activation
model.add(Dense(128, activation='relu'))
# Add the output layer with a single unit (assuming regression) and no activation
model.add(Dense(2))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Evaluation of Model
val_loss = model.evaluate(X_test, y_test)

# Predict using the trained model
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#Print Metrics to evaluate performance of Model
#We are looking to have the smalles val_loss and rmse - this would be our metrics to compare with other models
print("Validation loss of model tested:", val_loss)
print("Root Mean Squared Error (RMSE):", rmse)






# ---------------- #
#####
# CODE FOR LATER
# Once we have find the best tuned model, we save it, load it and tested with some images that the model has not seen yet.
# For this create a folder called "models" and other called "image_to_pred"
# Add in .gitignore to ignore this folders
#####
# from tensorflow.keras.models import load_model
# # Save the trained model to a .h5 file
# model.save('../models/"n_images"_model_"name_of_model".h5') #Change name of Model after saving, correct path if needed
# # Load the saved model
# saved_model = load_model('../models/500_model_VGG16.h5') #Change name of Model before loding, correct path if needed
# # Load and preprocess a new image for prediction
# new_image = Image.open('../image_to_pred/new_image2.jpg')  # Replace 'new_image.jpg' with the path to your new image
# new_image = new_image.resize((640, 480))  # Resize the image to match the input shape of the model
# new_image_array = img_to_array(new_image)
# new_image_array = new_image_array.astype('float32') / 255.0  # Normalize the pixel values between 0 and 1
## If an other preprocessing was added in the tunning of the model, need to be here as well.
# # Perform the prediction
# prediction = saved_model.predict(np.expand_dims(new_image_array, axis=0))
# # Display the prediction
# print("Predicted Latitude: ", prediction[0][0])
# print("Predicted Longitude: ", prediction[0][1])
# ---------------- #