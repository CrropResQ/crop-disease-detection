import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

print("Loading model...")
model = tf.keras.models.load_model("rice_disease_densenet.h5")
print("Model loaded successfully!")

class_names = [
    "Bacterialblight",
    "Blast",
    "Brownspot",
    "Tungro"
]

IMG_SIZE = 224

img_path = input("leaf_blast.jpg")

if not os.path.exists(img_path):
    print("Image file not found!")
    exit()

print("Loading image...")

img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

img_array = (img_array / 127.5) - 1

print("Predicting...")

predictions = model.predict(img_array)

predicted_class_index = np.argmax(predictions)
predicted_class_name = class_names[predicted_class_index]

confidence = np.max(predictions) * 100

print("\nPredicted Disease:", predicted_class_name)
print("Confidence: {:.2f}%".format(confidence))