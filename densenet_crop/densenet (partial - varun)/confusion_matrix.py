import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns

# LOAD MODEL
# model = tf.keras.models.load_model("rice_disease_mobilenet.h5")

model = tf.keras.models.load_model("rice_disease_densenet.h5")

# model = tf.keras.models.load_model("..//rice_disease_densenet.h5")

# DATASET PATH
dataset_path = "Rice_Leaf_AUG"

# LOAD VALIDATION DATASET
val_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# CLASS NAMES
class_names = val_dataset.class_names

# TRUE LABELS
y_true = np.concatenate([y for x, y in val_dataset], axis=0)

# PREDICTIONS
predictions = model.predict(val_dataset)

y_pred = np.argmax(predictions, axis=1)

# CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)

# PLOT
plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - MobileNetV2")

plt.savefig("confusion_matrix.png")

print("Confusion Matrix Saved Successfully")

# plt.show()