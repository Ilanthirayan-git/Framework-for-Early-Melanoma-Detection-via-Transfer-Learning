import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = tf.keras.models.load_model('poultry_disease_detection_densenet.h5')  # Replace with your .h5 file path

# Image parameters (same as used during training)
img_height, img_width = 224, 224

# Function to preprocess a single image
def preprocess_image(image_path):
    # Load the image
    img = load_img(image_path, target_size=(img_height, img_width))
    # Convert the image to an array
    img_array = img_to_array(img)
    # Rescale the image (same rescaling as during training)
    img_array = img_array / 255.0
    # Expand dimensions to match the input shape for the model (batch size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make a prediction on a new image
def predict_image(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)
    # Make a prediction
    prediction = model.predict(img_array)[0][0]  # Extract the prediction value
    
    # Print the raw prediction for debugging
    print(f'Raw prediction value: {prediction}')
    
    # Adjust the interpretation of the prediction
    if prediction >= 0.5:
        return "Predicted Class: Healthy"
    else:
        return "Predicted Class: Melanoma Affected"

# Test the prediction on a new image
image_path = './Dataset/test/Melanoma/AUG_0_1002.jpeg'  # Replace with the path to your image
prediction_result = predict_image(image_path)
print(prediction_result)

