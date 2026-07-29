import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121

# -----------------------------
# DATASET PATH
# -----------------------------
dataset_path = "Rice_Leaf_AUG"

# -----------------------------
# IMAGE SETTINGS
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 16

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
# PREFETCH
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# LOAD DENSENET
# -----------------------------
base_model = DenseNet121(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze pretrained layers
base_model.trainable = False

# -----------------------------
# BUILD MODEL
# -----------------------------
model = models.Sequential([

    layers.Rescaling(1./127.5, offset=-1),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu'),

    layers.Dropout(0.3),

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
# SUMMARY
# -----------------------------
model.summary()

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5
)

# -----------------------------
# EVALUATE
# -----------------------------
loss, accuracy = model.evaluate(validation_dataset)

print("\nValidation Accuracy:", accuracy * 100)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("rice_disease_densenet.h5")

print("\nDenseNet Model Saved Successfully")