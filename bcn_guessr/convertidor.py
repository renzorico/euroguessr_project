from PIL import Image
import os

folder_path_4_chan = "data/img_convertor/4_chan_img/"
folder_path_to_save = "data/img_convertor/img_converted/"
counter = 0

# Get a list of files in the folder (excluding hidden files)
image_files = [filename for filename in os.listdir(folder_path_4_chan) if not filename.startswith('.')]

for filename in image_files:
    #Get full file path
    image_path = os.path.join(folder_path_4_chan, filename)
    # Load the image with 4 channels (RGBA)
    image = Image.open(image_path)
    # Convert the image to RGB format (3 channels)
    image_rgb = image.convert("RGB")
    # Save the converted image as a new file
    image_rgb.save(os.path.join(folder_path_to_save, f"image_converted_to_rgb{counter}.png"))
    counter += 1
    
print("All images in folder path converted to RGB")