# Third Model, Using a Pretrained model ResNet50
# OPTIMAL MODEL _ OPTION 2

from utils import get_sample_dict
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the data
euro_dict = get_sample_dict()
images = np.array(euro_dict['images'])
targets = euro_dict['target']
targets = [target.split('_') for target in targets]
targets = [[float(coord) for coord in target] for target in targets]
targets = np.array(targets)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,  # Random rotation between -20 and 20 degrees
    width_shift_range=0.1,  # Random horizontal shift by 0.1
    height_shift_range=0.1,  # Random vertical shift by 0.1
    shear_range=0.2,  # Shear transformation with a maximum shear of 0.2
    zoom_range=0.2,  # Random zoom between 0.8 and 1.2
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill any newly created pixels due to rotation or shifting
)
datagen.fit(X_train)

# Load the ResNet50 model (excluding the top fully connected layers)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(320, 240, 3))

# Freeze some of the initial layers in the base model
for layer in base_model.layers[:160]:
    layer.trainable = False

# Create a new model by adding your own fully connected layers on top of the base model
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # Adding dropout regularization
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)  # Adding dropout regularization
x = Dense(64, activation='relu')(x)
predictions = Dense(2)(x)

# Define the combined model
model = Model(inputs=base_model.input, outputs=predictions)

# Lower learning rate
learning_rate = 0.0001

# Compile the model with a lower learning rate
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))

# Create an EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Increase the number of training epochs
epochs = 50

# Train the model with data augmentation
model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_test, y_test),
    epochs=epochs,
    callbacks=[early_stopping]
)

# Evaluation of Model
val_loss = model.evaluate(X_test, y_test)

# Predict using the trained model
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print Metrics to evaluate performance of Model
print("Validation loss of model tested:", val_loss)
print("Root Mean Squared Error (RMSE):", rmse)



# ---------------- #
#####
# 
# Once we have find the best tuned model, we save it, load it and tested with some images that the model has not seen yet.
# For this create a folder called "models" and other called "image_to_pred"
# Add in .gitignore to ignore this folders
#####
from tensorflow.keras.models import load_model
# Save the trained model to a .h5 file
model.save('/mnt/mydisk/50_resnet50_opt5.keras') #Change name of Model after saving, correct path if needed
# Load the saved model
# saved_model = load_model('resnet50_opt5.h5') #Change name of Model before loding, correct path if needed
# # Load and preprocess a new image for prediction
# from tensorflow.keras.utils import img_to_array
# from PIL import Image

# new_image = Image.open('new_image.jpg')  # Replace 'new_image.jpg' with the path to your new image
# new_image = new_image.resize((640, 480))  # Resize the image to match the input shape of the model
# new_image_array = img_to_array(new_image)
# new_image_array = new_image_array.astype('float32') / 255.0  # Normalize the pixel values between 0 and 1
# # If an other preprocessing was added in the tunning of the model, need to be here as well.
# # Perform the prediction
# prediction = saved_model.predict(np.expand_dims(new_image_array, axis=0))
# # Display the prediction
# print("Predicted Latitude: ", prediction[0][0])
# print("Predicted Longitude: ", prediction[0][1])
# ---------------- #