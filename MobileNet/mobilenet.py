import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# -----------------------------
# DATASET PATH
# -----------------------------
dataset_path = "DATASETS/Rice_Leaf_AUG"

# -----------------------------
# IMAGE SETTINGS
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32

# -----------------------------
# LOAD DATASET
# -----------------------------
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# -----------------------------
# CLASS NAMES
# -----------------------------
class_names = train_dataset.class_names

print("\nClasses Found:")
print(class_names)

# -----------------------------
# PREFETCHING
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# LOAD MOBILENET MODEL
# -----------------------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze pretrained layers
base_model.trainable = False
# base_model.trainable = True

# # Freeze most layers
# for layer in base_model.layers[:-30]:
#     layer.trainable = False



# -----------------------------
# BUILD MODEL
# -----------------------------
model = models.Sequential([
    layers.Rescaling(1./127.5, offset=-1),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu'),

    layers.Dense(len(class_names), activation='softmax')
])

# -----------------------------
# COMPILE MODEL
# -----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# MODEL SUMMARY
# -----------------------------
model.summary()

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=4
)

# -----------------------------
# EVALUATE MODEL
# -----------------------------
loss, accuracy = model.evaluate(validation_dataset)

print("\nValidation Accuracy:", accuracy * 100)


# -----------------------------
# ACCURACY GRAPH
# -----------------------------
plt.figure()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('MobileNetV2 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend(['Training', 'Validation'])

plt.savefig("mobilenet_accuracy.png")

print("Accuracy Graph Saved")

# -----------------------------
# LOSS GRAPH
# -----------------------------
plt.figure()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('MobileNetV2 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend(['Training', 'Validation'])

plt.savefig("mobilenet_loss.png")

print("Loss Graph Saved")




model.save("rice_disease_mobilenet.h5")

print("\nModel Saved Successfully")