# from flask import Flask, render_template, request, redirect, url_for
# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import numpy as np

# app = Flask(__name__)

# # Load the trained model
# model = tf.keras.models.load_model('poultry_disease_detection_densenet.h5')  # Replace with your .h5 file path

# # Image parameters (same as used during training)
# img_height, img_width = 224, 224

# # Function to preprocess a single image
# def preprocess_image(image_path):
#     img = load_img(image_path, target_size=(img_height, img_width))
#     img_array = img_to_array(img)
#     img_array = img_array / 255.0  # Rescale
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     return img_array

# # Function to make a prediction on a new image
# def predict_image(image_path):
#     img_array = preprocess_image(image_path)
#     prediction = model.predict(img_array)[0][0]  # Get prediction score
#     return prediction

# # Home route to render the upload form
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route to handle image upload and prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return redirect(url_for('index'))
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return redirect(url_for('index'))
    
#     # Save the uploaded file
#     if file:
#         image_path = os.path.join('uploads', file.filename)
#         file.save(image_path)
        
#         # Make a prediction
#         prediction_score = predict_image(image_path)
        
#         # Determine the result based on the prediction score
#         if prediction_score >= 0.5:
#             result = "Healthy"
#         else:
#             result = "Melanoma Affected"
        
#         # Send the result back to the frontend
#         return render_template('result.html', prediction=result, image_file=file.filename)

# if __name__ == "__main__":
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(debug=True)



#//http://127.0.0.1:5000/


# Newly only confusion matrix 

# from flask import Flask, render_template, request, redirect, url_for
# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import numpy as np
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# app = Flask(__name__)

# # Load the trained model
# model = tf.keras.models.load_model('poultry_disease_detection_densenet.h5')  # Replace with your .h5 file path

# # Image parameters (same as used during training)
# img_height, img_width = 224, 224

# # Store true labels and predictions for confusion matrix
# true_labels = []
# predicted_labels = []

# # Function to preprocess a single image
# def preprocess_image(image_path):
#     img = load_img(image_path, target_size=(img_height, img_width))
#     img_array = img_to_array(img)
#     img_array = img_array / 255.0  # Rescale
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     return img_array

# # Function to make a prediction on a new image
# def predict_image(image_path):
#     img_array = preprocess_image(image_path)
#     prediction = model.predict(img_array)[0][0]  # Get prediction score
#     return prediction

# # Function to plot confusion matrix
# def plot_confusion_matrix(true_labels, predicted_labels):
#     cm = confusion_matrix(true_labels, predicted_labels)
#     plt.figure(figsize=(6,6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Melanoma Affected', 'Healthy'], yticklabels=['Melanoma Affected', 'Healthy'])
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#     plt.savefig('static/confusion_matrix.png')
#     plt.close()

# # Home route to render the upload form
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route to handle image upload and prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return redirect(url_for('index'))
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return redirect(url_for('index'))
    
#     # Save the uploaded file
#     if file:
#         image_path = os.path.join('uploads', file.filename)
#         file.save(image_path)
        
#         # Make a prediction
#         prediction_score = predict_image(image_path)
        
#         # Determine the result based on the prediction score
#         if prediction_score >= 0.5:
#             result = "Healthy"
#             predicted_label = 1
#         else:
#             result = "Melanoma Affected"
#             predicted_label = 0
        
#         # Assume that the true label is provided in the filename (e.g., melanoma_1.jpg)
#         true_label = 1 if 'healthy' in file.filename.lower() else 0
        
#         # Store true and predicted labels
#         true_labels.append(true_label)
#         predicted_labels.append(predicted_label)
        
#         # Plot and save the confusion matrix
#         plot_confusion_matrix(true_labels, predicted_labels)
        
#         # Send the result back to the frontend
#         return render_template('result.html', prediction=result, image_file=file.filename, confusion_matrix='static/confusion_matrix.png')

# if __name__ == "__main__":
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(debug=True)


# accuracy threhsoold

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = tf.keras.models.load_model('poultry_disease_detection_densenet.h5')  # Replace with your .h5 file path

# Image parameters (same as used during training)
img_height, img_width = 224, 224

# Function to preprocess a single image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make a prediction on a new image
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)[0][0]  # Get prediction score
    return prediction

# Function to plot confusion matrix for a single prediction
def plot_confusion_matrix(true_label, predicted_label):
    cm = confusion_matrix([true_label], [predicted_label])
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Melanoma Affected', 'Healthy'], 
                yticklabels=['Melanoma Affected', 'Healthy'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the confusion matrix plot to the static folder
    plt.savefig('static/confusion_matrix.png')
    plt.close()

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Home route to render the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    # Save the uploaded file
    if file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        
        # Make a prediction
        prediction_score = predict_image(image_path)
        
        # Determine the result based on the prediction score
        if prediction_score >= 0.5:
            predicted_label = 1  # Healthy
            result = False  # Output False for Healthy
        else:
            predicted_label = 0  # Melanoma Affected
            result = True  # Output True for Melanoma Affected
        
        # Assume that the true label is provided in the filename (e.g., healthy_1.jpg or melanoma_1.jpg)
        true_label = 1 if 'healthy' in file.filename.lower() else 0
        
        # Plot and save the confusion matrix for the current prediction
        plot_confusion_matrix(true_label, predicted_label)

        # Send the result (True/False) and confusion matrix image back to the frontend
        return render_template('result.html', 
                               prediction=result, 
                               image_file=file.filename, 
                               confusion_matrix='static/confusion_matrix.png')

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)





