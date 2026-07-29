import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =====================================
# LOAD MODEL
# =====================================

model = tf.keras.models.load_model(
    "densenet_rice_leaf_model.h5"
)

# =====================================
# DATASET PATH
# =====================================

dataset_path = "../MobileNet/Rice_Leaf_AUG"

# =====================================
# IMAGE PREPROCESSING
# =====================================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# =====================================
# PREDICTIONS
# =====================================

predictions = model.predict(test_data)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_data.classes

class_labels = list(test_data.class_indices.keys())

# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

# =====================================
# PLOT GRAPH
# =====================================

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.title("Confusion Matrix - DenseNet")

plt.show()