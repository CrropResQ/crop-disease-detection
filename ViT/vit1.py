import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from collections import Counter

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import tensorflow as tf

from transformers import TFViTModel, ViTImageProcessor
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)
processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)

inputs = processor(
    images=image,
    return_tensors="tf"
)



# Project Root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from evaluation import evaluate_model

# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(
    BASE_DIR,
    "..",
    "DATASETS",
    "Rice_Leaf_AUG"
)

IMG_SIZE = (224, 224)

BATCH_SIZE = 16

EPOCHS = 25

LEARNING_RATE = 0.0005

MODEL_NAME = "ViT"

model.save_weights("rice_vit.weights.h5")
model.load_weights("rice_vit.weights.h5")


# ============================================================
# Data Preprocessing
# ============================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: x / 255.0,


    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,

    validation_split=0.2
)
val_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: x / 255.0,
    validation_split=0.2
)



# ============================================================
# Load Dataset
# ============================================================

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)
# ============================================================
# Dataset Information
# ============================================================

num_classes = train_data.num_classes

print(f"\nDetected Classes: {num_classes}")
print(train_data.class_indices)

with open("class_indices.pkl", "wb") as f:
    pickle.dump(train_data.class_indices, f)

print("\nClass Distribution:")
print(Counter(train_data.classes))

# ============================================================
# Load Pretrained Vision Transformer
# ============================================================

vit_model = TFViTModel.from_pretrained(
    "google/vit-base-patch16-224"
)

vit_model.trainable = True

# Freeze most transformer blocks
vit_model.trainable = True


# ============================================================
# Build Model
# ============================================================

inputs = Input(shape=(224, 224, 3))

x = inputs / 255.0

vit_outputs = vit_model(pixel_values=x)

cls_token = vit_outputs.last_hidden_state[:, 0]



x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(0.5)(x)

outputs = Dense(
    num_classes,
    activation="softmax"
)(x)

model = Model(inputs, outputs)


# ============================================================
# Compile Model
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

with open("model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

print("Model summary saved.")

# ============================================================
# Callbacks
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    MODEL_FILE,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ============================================================
# Train Model
# ============================================================

print("\nStarting Training...\n")

start_time = time.time()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[
        early_stop,
        checkpoint
    ]
)

training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds")

# ============================================================
# Save Training History
# ============================================================

with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("Training history saved.")

# ============================================================
# Evaluate Model
# ============================================================

model = load_model(MODEL_FILE)

loss, accuracy = model.evaluate(val_data)

print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy*100:.2f}%")

evaluate_model(
    model=model,
    history=history,
    val_data=val_data,
    model_name=MODEL_NAME,
    training_time=training_time,
    model_path=MODEL_FILE
)

print("Evaluation completed.")

