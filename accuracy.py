# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import load_model
# from tensorflow.keras.optimizers import Adam
# import matplotlib.pyplot as plt

# # Dataset paths (ensure these paths are correct)
# train_dir = 'Dataset/train'
# test_dir = 'Dataset/test'

# # Image parameters
# img_height, img_width = 224, 224
# batch_size = 32

# # Data augmentation for training (same as you used during model training)
# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     rotation_range=40,
#     width_shift_range=0.4,
#     height_shift_range=0.4,
#     shear_range=0.4,
#     zoom_range=0.4,
#     horizontal_flip=True,
#     vertical_flip=True,
#     fill_mode='nearest'
# )

# # Test data should only be rescaled
# test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# # Create data generators
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary'
# )

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary',
#     shuffle=False
# )

# # Load your saved model
# model = load_model('poultry_disease_detection_densenet.h5')

# # Recompile the model with the same settings used during training
# model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# # Re-evaluate the model on training and validation datasets to get accuracies
# train_loss, train_accuracy = model.evaluate(train_generator)
# test_loss, test_accuracy = model.evaluate(test_generator)

# print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
# print(f'Validation Accuracy: {test_accuracy * 100:.2f}%')

# # Plot Training vs Validation Accuracy
# accuracies = [train_accuracy, test_accuracy]
# labels = ['Training Accuracy', 'Validation Accuracy']

# plt.figure(figsize=(6, 4))
# plt.bar(labels, accuracies, color=['blue', 'green'])
# plt.ylabel('Accuracy')
# plt.title('Training Accuracy vs Validation Accuracy')
# plt.show()

#---------------------------------------------Performance Metrics (Accuracy,Precision,Recall)
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset paths (ensure these paths are correct)
train_dir = 'Dataset/train'
test_dir = 'Dataset/test'

# Image parameters
img_height, img_width = 224, 224
batch_size = 32

# Data augmentation for training (same as you used during model training)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Create data generators
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Load your saved model
model = load_model('poultry_disease_detection_densenet.h5')

# Get true labels from the test generator
true_classes = test_generator.classes

# Predict probabilities
predictions = model.predict(test_generator)

# Convert probabilities to binary predictions
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes)
recall = recall_score(true_classes, predicted_classes)
f1 = f1_score(true_classes, predicted_classes)

# Display the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Generate classification report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes))

# Generate confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices, yticklabels=test_generator.class_indices)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()
