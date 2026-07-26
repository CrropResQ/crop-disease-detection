import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing import image
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)

# ---------------------------------
# CONFIG
# ---------------------------------

MODEL_PATH = "rice_disease_mobilenet.h5"

DATASET_PATH = "../DATASETS/Rice_Leaf_AUG"

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 123

# ---------------------------------
# LOAD MODEL
# ---------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------------------
# CLASS NAMES
# ---------------------------------

class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Sheath Blight"
]

# ==========================================================
# SINGLE IMAGE PREDICTION
# ==========================================================

def predict_single(img_path):

    print("\nLoading Image :", img_path)

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    # Same preprocessing used during training
    img = img / 127.5 - 1

    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    confidence = prediction[0][predicted_index] * 100

    print("\n========================================")
    print("      CROP DISEASE PREDICTION")
    print("========================================")

    print(f"\nPredicted Disease : {predicted_class}")
    print(f"Confidence        : {confidence:.2f}%")

    print("\n----------------------------------------")
    print("Class Probabilities")
    print("----------------------------------------")

    for i, name in enumerate(class_names):
        print(f"{name:25s}: {prediction[0][i]*100:.2f}%")

    print("========================================")


# ==========================================================
# DATASET EVALUATION
# ==========================================================

def evaluate_dataset():

    print("\nLoading Test Dataset...\n")

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.30,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = test_ds.map(
        lambda x, y: ((x / 127.5) - 1, y)
    )

    print("\nPredicting...\n")

    predictions = model.predict(test_ds, verbose=1)

    y_pred = np.argmax(predictions, axis=1)

    y_true = np.concatenate(
        [labels.numpy() for _, labels in test_ds],
        axis=0
    )

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro"
    )

    print("\n")
    print("=" * 65)
    print("MODEL EVALUATION")
    print("=" * 65)

    print(f"\nAccuracy        : {accuracy*100:.2f}%")
    print(f"Macro Precision : {precision:.4f}")
    print(f"Macro Recall    : {recall:.4f}")
    print(f"Macro F1 Score  : {f1:.4f}")

    print("\n")
    print("=" * 65)
    print("CLASSIFICATION REPORT")
    print("=" * 65)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    print("\n")
    print("=" * 65)
    print("CONFUSION MATRIX")
    print("=" * 65)

    print(cm)

    plt.figure(figsize=(9,7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)

    print("\nConfusion matrix saved as confusion_matrix.png")

    print("\nEvaluation Completed Successfully.\n")


# ==========================================================
# MAIN
# ==========================================================

if len(sys.argv) > 1:

    if sys.argv[1].lower() == "evaluate":
        evaluate_dataset()

    else:
        predict_single(sys.argv[1])

else:
    predict_single("healthy.jpg")