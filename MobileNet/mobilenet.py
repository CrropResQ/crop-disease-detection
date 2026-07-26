import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================================
# CONFIGURATION
# ======================================================

DATASET_PATH = "../DATASETS/Rice_Leaf_AUG"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
SEED = 123

MODEL_DIR = "models"
PLOT_DIR = "plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ======================================================
# LOAD DATASET
# ======================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names

print("\n===================================")
print("Classes Found")
print("===================================")

for i, cls in enumerate(class_names):
    print(f"{i} : {cls}")

print("===================================\n")

# ======================================================
# PERFORMANCE OPTIMIZATION
# ======================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# ======================================================
# DATA AUGMENTATION
# ======================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
])

# ======================================================
# LOAD MOBILENETV2
# ======================================================

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze all layers
base_model.trainable = False

# ======================================================
# BUILD MODEL
# ======================================================

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

x = layers.Dense(256, activation="relu")(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# ======================================================
# COMPILE
# ======================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================================================
# CALLBACKS
# ======================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "mobilenet.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        verbose=1
    )

]

# ======================================================
# TRAIN
# ======================================================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ======================================================
# FINAL VALIDATION
# ======================================================

loss, accuracy = model.evaluate(validation_dataset)

print("\n===================================")
print(f"Validation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy*100:.2f}%")
print("===================================")

# ======================================================
# ACCURACY GRAPH
# ======================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("MobileNetV2 Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_DIR, "accuracy.png"),
    dpi=300
)

plt.close()

# ======================================================
# LOSS GRAPH
# ======================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("MobileNetV2 Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_DIR, "loss.png"),
    dpi=300
)

plt.close()

print("\nGraphs saved successfully.")

print("Model saved successfully.")

print("\nTraining Complete.")