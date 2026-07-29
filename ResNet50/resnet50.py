import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
from collections import Counter

import os #added by varun to imoprt dataset from relative folder

# Dataset path
# dataset_path = r"C:\Users\monke\crop-disease-detection\MobileNet\Rice_Leaf_AUG"

#Dataset path to the one that exists in the other relative folder (edited by varun)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "..", "DATASETS", "Rice_Leaf_AUG")



# Image settings
IMG_SIZE = (224, 224)  
BATCH_SIZE = 32

# Data preprocessing
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# Training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Number of classes
num_classes = train_data.num_classes

print(f"\nDetected Classes: {num_classes}")
print(train_data.class_indices)

# Check dataset balance
print("\nClass Distribution:")
print(Counter(train_data.classes))

# Load ResNet50 base model
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze pretrained layers
base_model.trainable = False

# Build model
model = Sequential([
    base_model,

    # IMPORTANT FIX
    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Evaluate model
loss, accuracy = model.evaluate(val_data)

print(f"\nValidation Accuracy: {accuracy*100:.2f}%")

# Predictions
predictions = model.predict(val_data)
predicted_classes = np.argmax(predictions, axis=1)

# True labels
true_classes = val_data.classes

# Class labels
class_labels = list(val_data.class_indices.keys())

# Classification report
print("\nClassification Report:\n")

print(classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels
))

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - ResNet50")

plt.show()

# Accuracy graph
plt.figure(figsize=(8,6))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend(['Train', 'Validation'])

plt.show()

# Loss graph
plt.figure(figsize=(8,6))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend(['Train', 'Validation'])

plt.show()

# Save model
model.save("rice_resnet50.keras")

print("\nModel saved successfully.")
