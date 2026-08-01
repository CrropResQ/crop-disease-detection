import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import tensorflow as tf

# ------------------------------------------------------------
# ViT backbone
# ------------------------------------------------------------
# `vit-keras` is no longer usable: it hard-imports `tensorflow_addons`,
# which reached end-of-life in May 2024 and stopped supporting current
# TensorFlow/Keras versions (its repo is now archived). We use `keras_hub`
# instead -- Keras's own, actively maintained model hub, which ships a
# native ViT implementation with ImageNet-pretrained weights and no
# tensorflow_addons dependency.
#
#   pip install --upgrade keras keras-hub tensorflow
#
import keras
import keras_hub

from keras import layers
from keras.models import load_model


from keras.optimizers import Adam, SGD, RMSprop

from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# Project Root
# NOTE: __file__ is only defined when running as a script (python script.py).
# In notebooks / interactive cells (Jupyter, VS Code interactive window) it
# doesn't exist, so we fall back to the current working directory instead.
# __file__ points at a FILE, so BASE_DIR = dirname(__file__).
# cwd already IS a directory, so in the fallback case BASE_DIR = cwd directly
# (no extra dirname(), or you'd climb one level too high).
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath(os.getcwd())

PROJECT_ROOT = os.path.dirname(BASE_DIR)

sys.path.insert(0, PROJECT_ROOT)

from evaluation import evaluate_model

# ============================================================
# Configuration
# ============================================================

dataset_path = os.path.join(
    BASE_DIR,
    "..",
    "DATASETS",
    "Rice_Leaf_AUG"
)

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 25

LEARNING_RATE = 1e-5

# One of: "adam", "sgd", "rmsprop"
# Change this per the hyperparameter tuning plan (experiments 7-9 sweep
# the optimizer while LR / batch size stay fixed at their best values).
OPTIMIZER_NAME = "adam"

MODEL_NAME = "ViT-B16"

MODEL_FILE = "rice_vit_b16.keras"

# keras_hub preset name for ViT-B16 @ 224x224, ImageNet-pretrained.
# See https://keras.io/keras_hub/api/models/vit/vit_backbone/ for the
# full preset list (e.g. swap to "vit_base_patch16_224_imagenet21k" if
# you want the ImageNet-21k-only backbone instead).
VIT_PRESET = "vit_base_patch16_224_imagenet"

# Number of trailing backbone layers to keep trainable during fine-tuning
# (mirrors the "freeze early layers" strategy used for MobileNetV2 /
# ResNet50 elsewhere in this study, where the last 30 layers are trainable).
UNFROZEN_LAYERS = 4


def get_optimizer(name, lr):
    name = name.lower()
    if name == "adam":
        return Adam(learning_rate=lr)
    elif name == "sgd":
        return SGD(learning_rate=lr, momentum=0.9)
    elif name == "rmsprop":
        return RMSprop(learning_rate=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ============================================================
# Load Dataset (tf.data)
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# ============================================================
# Dataset Information
# ============================================================

class_names = train_ds.class_names
num_classes = len(class_names)

class_indices = {name: idx for idx, name in enumerate(class_names)}

print(f"\nDetected Classes: {num_classes}")
print(class_indices)

with open("class_indices.pkl", "wb") as f:
    pickle.dump(class_indices, f)

# ============================================================
# Data Augmentation
# ============================================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.20),
    layers.RandomTranslation(0.20, 0.20),
])

normalization = layers.Rescaling(
    scale=1/127.5,
    offset=-1
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    train_ds
    .map(
        lambda x, y: (
            normalization(data_augmentation(x, training=True)),
            y
        ),
        num_parallel_calls=AUTOTUNE
    )
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds
    .map(
        lambda x, y: (
            normalization(x),
            y
        ),
        num_parallel_calls=AUTOTUNE
    )
    .prefetch(AUTOTUNE)
)

backbone = keras_hub.models.ViTBackbone.from_preset(VIT_PRESET)

backbone.trainable = False

print("\n===== Backbone Layers =====")
for i, layer in enumerate(backbone.layers):
    print(i, layer.name, type(layer))
print("===========================\n")

model = keras_hub.models.ViTImageClassifier(
    backbone=backbone,
    num_classes=num_classes,
    preprocessor=None,
    pooling="gap",
    intermediate_dim=256,
    activation="softmax",
    dropout=0.5,
)

# ============================================================
# Build Model
# ============================================================
# ViTImageClassifier wraps the backbone with a pooling + Dense head.
# pooling="gap" averages over all patch tokens (the same role
# GlobalAveragePooling1D played over the ViT's token sequence in the
# vit-keras version), rather than using only the [CLS] token.
# preprocessor=None because preprocessing is already handled by
# ImageDataGenerator above.

model = keras_hub.models.ViTImageClassifier(
    backbone=backbone,
    num_classes=num_classes,
    preprocessor=None,
    pooling="gap",
    intermediate_dim=256,
    activation="softmax",
    dropout=0.5,
)

# ============================================================
# Compile Model
# ============================================================

model.compile(
    optimizer=get_optimizer(OPTIMIZER_NAME, LEARNING_RATE),
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
    train_ds,
    validation_data=val_ds,
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
# keras_hub registers its custom layers/classes with Keras's saving
# registry on import, so as long as `keras_hub` has been imported
# (it has, above), load_model needs no custom_objects argument.

model = load_model(MODEL_FILE)

model.compile(
    optimizer=get_optimizer(OPTIMIZER_NAME, LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

loss, accuracy = model.evaluate(val_ds)

print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy*100:.2f}%")

evaluate_model(
    model=model,
    history=history,
    val_data=val_ds,
    model_name=MODEL_NAME,
    training_time=training_time,
    model_path=MODEL_FILE
)

print("Evaluation completed.")