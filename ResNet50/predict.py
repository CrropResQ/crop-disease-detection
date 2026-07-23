import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

import numpy as np

# Load model
model = load_model("ResNet50/rice_resnet50.keras")

# Class labels
class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight"
]

# Image path
img_path = r"test_images/blbb.jpg"

# Load image
img = image.load_img(img_path, target_size=(224,224))

# Convert image to array
img_array = image.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Preprocess image
img_array = preprocess_input(img_array)

# Predict
predictions = model.predict(img_array)

# Get highest probability class
predicted_index = np.argmax(predictions)

# Get label
predicted_label = class_names[predicted_index]

# Confidence
confidence = np.max(predictions) * 100

# Print results
print("\n==============================")
print("Rice Leaf Disease Prediction")
print("==============================")

print(f"\nPredicted Disease : {predicted_label}")
print(f"Confidence         : {confidence:.2f}%")