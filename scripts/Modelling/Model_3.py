# Third Model, Using a Pretrained model ReNet50

from utils import get_sample_dict
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
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

# Load the ResNet50 model (excluding the top fully connected layers)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(480, 640, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Create a new model by adding your own fully connected layers on top of the base model
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
predictions = Dense(2)(x)

# Define the combined model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(loss='mean_squared_error', optimizer=Adam())

# Create an EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, callbacks=[early_stopping])

# Evaluation of Model
val_loss = model.evaluate(X_test, y_test)

# Predict using the trained model
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print Metrics to evaluate performance of Model
# We are looking to have the smallest val_loss and rmse - these would be our metrics to compare with other models
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