
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import ResNet50, DenseNet121
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import os

# # Dataset paths
# train_dir = 'Dataset/train'
# test_dir = 'Dataset/test'

# # Image parameters
# img_height, img_width = 224, 224
# batch_size = 32

# # Data augmentation for training
# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     rotation_range=30,  # Increased rotation
#     width_shift_range=0.3,  # Increased shift
#     height_shift_range=0.3,
#     shear_range=0.3,  # Increased shear
#     zoom_range=0.3,  # Increased zoom
#     horizontal_flip=True,
#     vertical_flip=True,  # Added vertical flip
#     fill_mode='nearest'
# )

# # Test data should only be rescaled
# test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# # Create data generators
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary'  # Use 'categorical' for multi-class classification
# )

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary',
#     shuffle=False
# )

# # Build a generic function for ResNet or DenseNet
# def build_model(model_type='resnet'):
#     if model_type == 'resnet':
#         base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
#     elif model_type == 'densenet':
#         base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

#     # Adding custom layers on top of the base model
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dropout(0.5)(x)  # Dropout to prevent overfitting
#     x = Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification (use 'softmax' for multi-class)

#     model = Model(inputs=base_model.input, outputs=x)

#     # Freezing some layers of the base model for transfer learning
#     for layer in base_model.layers[:100]:  # Freeze the first 100 layers
#         layer.trainable = False

#     # Compile the model with a reduced learning rate for fine-tuning
#     model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
#     return model

# # Choose either 'resnet' or 'densenet'
# model_type = 'densenet'  # Try with 'resnet' and 'densenet'
# model = build_model(model_type=model_type)

# # Training parameters
# epochs = 10

# # Callbacks: Early stopping and learning rate reduction on plateau
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# # Train the model
# model.fit(
#     train_generator,
#     epochs=epochs,
#     validation_data=test_generator,
#     callbacks=[early_stopping, reduce_lr]
# )

# # Evaluate the model
# loss, accuracy = model.evaluate(test_generator)
# print(f'Test Accuracy: {accuracy * 100:.2f}% using {model_type}')

# # Optionally, save the model
# model.save(f'poultry_disease_detection_{model_type}.h5')

#---------------------------------------------


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset paths
train_dir = 'Dataset/train'
test_dir = 'Dataset/test'

# Image parameters
img_height, img_width = 224, 224
batch_size = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Test data should only be rescaled
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Build the model
def build_model(model_type='densenet'):
    if model_type == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    elif model_type == 'densenet':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)

    # Freeze layers for transfer learning
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Choose model type
model_type = 'densenet'
model = build_model(model_type=model_type)

# Training parameters
epochs = 10  # Increased epochs for better learning

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {accuracy * 100:.2f}% using {model_type}')

# Predictions
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# True classes
true_classes = test_generator.classes

# Generate confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices, yticklabels=test_generator.class_indices)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print(classification_report(true_classes, predicted_classes))

# Save the model
model.save(f'poultry_disease_detection_{model_type}.h5')
