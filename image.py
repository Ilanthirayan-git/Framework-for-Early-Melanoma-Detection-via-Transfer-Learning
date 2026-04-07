import os
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot and save the confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels):
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Create the static directory if it doesn't exist
    static_dir = 'E:\\FinalYearProject\\static'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    # Plot the confusion matrix
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Melanoma Affected', 'Healthy'], 
                yticklabels=['Melanoma Affected', 'Healthy'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the confusion matrix as an image
    plt.savefig(os.path.join(static_dir, 'confusion_matrix.png'))  # Save in static folder
    plt.close()

# Example usage
# 0 = Melanoma Affected, 1 = Healthy
true_labels = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]  # Actual labels
predicted_labels = [0, 1, 0, 0, 1, 0, 1, 1, 1, 0]  # Predicted labels

# Generate the confusion matrix and save it
plot_confusion_matrix(true_labels, predicted_labels)

