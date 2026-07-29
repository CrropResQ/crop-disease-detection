import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# =====================================
# DATASET PATH
# =====================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "..", "Rice_Leaf_AUG")
# =====================================
# IMAGE SETTINGS
# =====================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

# =====================================
# DATA PREPROCESSING
# =====================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Training Data
train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Validation Data
test_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# =====================================
# LOAD DENSENET MODEL
# =====================================

base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# =====================================
# ADD CUSTOM LAYERS
# =====================================

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)

predictions = Dense(
    train_data.num_classes,
    activation='softmax'
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

# =====================================
# COMPILE MODEL
# =====================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =====================================
# MODEL SUMMARY
# =====================================

model.summary()

# =====================================
# TRAIN MODEL
# =====================================

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

# =====================================
# SAVE MODEL
# =====================================

model.save("densenet_rice_leaf_model.h5")

print("\nModel Saved Successfully!")

# =====================================
# PLOT ACCURACY GRAPH
# =====================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("DenseNet Rice Leaf Disease Accuracy")
plt.legend()

plt.show()

# =====================================
# PLOT LOSS GRAPH
# =====================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("DenseNet Rice Leaf Disease Loss")
plt.legend()

plt.show()