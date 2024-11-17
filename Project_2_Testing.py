import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from google.colab import drive
drive.mount('/content/drive')

'''Test Results for Model 1'''

# Load the model
model1 = tf.keras.models.load_model('/content/drive/MyDrive/Notability/Datasets/model1.keras')

# List of the class labels
class_labels = ['Crack', 'Missing Head', 'Paint-off']

# Paths for the test images
image_paths = ['/content/drive/MyDrive/Notability/Datasets/Data/test/crack/test_crack.jpg',
               '/content/drive/MyDrive/Notability/Datasets/Data/test/missing-head/test_missinghead.jpg',
               '/content/drive/MyDrive/Notability/Datasets/Data/test/paint-off/test_paintoff.jpg']

# Load and process the image
def preprocess_image(image_path, target_size = (96, 96)):
    img = load_img(image_path, target_size = target_size, color_mode = "grayscale")
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis = 0)
    img_array /= 255.0
    return img_array

# Setting true labels
true_labels = {
    '/content/drive/MyDrive/Notability/Datasets/Data/test/crack/test_crack.jpg': 'Crack',
    '/content/drive/MyDrive/Notability/Datasets/Data/test/paint-off/test_paintoff.jpg': 'Paint-off',
    '/content/drive/MyDrive/Notability/Datasets/Data/test/missing-head/test_missinghead.jpg': 'Missing Head'
}

# Function to display the images with prediction probabilities
def display_prediction(image_path):
    img_array = preprocess_image(image_path)
    predictions = model1.predict(img_array)
    predicted_probs = predictions[0] * 100  # Convert to percentage
    predicted_label = class_labels[np.argmax(predicted_probs)]

    # Get the true label
    true_label = true_labels[image_path]

    # Load original image for display
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Display the image with probabilities
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    plt.title(f"True Class Classification Label: {true_label}\nPredicted Class Classification Label: {predicted_label}")

    # Overlay each probability on the image
    text_y_position = 30
    spacing = 35
    for i, (label, prob) in enumerate(zip(class_labels, predicted_probs)):
        plt.text(10, text_y_position, f"{label}: {prob:.1f}%", color='green', fontsize = 12)
        text_y_position += spacing  # Move down for the next class

    plt.show()

# Iterate through each image path and display predictions
for image_path in image_paths:
    display_prediction(image_path)
    
    
'''Test Results for Model 2'''

model2 = tf.keras.models.load_model('/content/drive/MyDrive/Notability/Datasets/model2.keras')

def display_prediction(image_path):
    img_array = preprocess_image(image_path)
    predictions = model2.predict(img_array)
    predicted_probs = predictions[0] * 100  # Convert to percentage
    predicted_label = class_labels[np.argmax(predicted_probs)]

    # Get the true label
    true_label = true_labels[image_path]

    # Load original image for display
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Display the image with probabilities
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    plt.title(f"True Class Classification Label: {true_label}\nPredicted Class Classification Label: {predicted_label}")

    # Overlay each probability on the image
    text_y_position = 30
    spacing = 35
    for i, (label, prob) in enumerate(zip(class_labels, predicted_probs)):
        plt.text(10, text_y_position, f"{label}: {prob:.1f}%", color='green', fontsize = 12)
        text_y_position += spacing  # Move down for the next class

    plt.show()

# Iterate through each image path and display predictions
for image_path in image_paths:
    display_prediction(image_path)