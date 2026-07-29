import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# =====================================
# LOAD TRAINED MODEL
# =====================================

model = tf.keras.models.load_model(
    "densenet_rice_leaf_model.h5"
)

# =====================================
# CLASS NAMES
# =====================================

class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight"
]

# =====================================
# LOAD TEST IMAGE
# =====================================

img_path = "leaf2.jpg"

img = image.load_img(
    img_path,
    target_size=(224, 224)
)

img_array = image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

# =====================================
# PREDICTION
# =====================================

prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]

confidence = np.max(prediction) * 100

# =====================================
# OUTPUT
# =====================================

print("\n===================================")
print(" Rice Leaf Disease Prediction")
print("===================================\n")

print("Predicted Disease :", predicted_class)

print("Confidence        :", round(confidence, 2), "%")