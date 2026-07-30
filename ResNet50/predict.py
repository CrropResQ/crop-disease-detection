import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

import numpy as np
import pickle


# Load model
model = load_model("rice_resnet50.keras")


# Load class names
with open("class_indices.pkl", "rb") as f:
    class_indices = pickle.load(f)

class_names = list(class_indices.keys())


# Image path
img_path = r"test_images\BLB.jpg"


# Load image
img = image.load_img(
    img_path,
    target_size=(224,224)
)


# Convert image
img_array = image.img_to_array(img)


# Add batch dimension
img_array = np.expand_dims(
    img_array,
    axis=0
)


# Preprocess
img_array = preprocess_input(img_array)


# Predict
predictions = model.predict(img_array)


predicted_index = np.argmax(predictions)

predicted_label = class_names[predicted_index]

confidence = np.max(predictions) * 100


print("\n==============================")
print("Rice Leaf Disease Prediction")
print("==============================")

print(f"\nPredicted Disease : {predicted_label}")
print(f"Confidence        : {confidence:.2f}%")