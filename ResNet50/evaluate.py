import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import classification_report
import numpy as np

# Load trained ResNet50 model
model = load_model("ResNet50/rice_resnet50.keras")

# Evaluation dataset path
dataset_path = "evaluation_dataset"

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Image preprocessing
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Load evaluation dataset
eval_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Predict on evaluation dataset
predictions = model.predict(eval_data)

# Convert predictions to class indices
predicted_classes = np.argmax(predictions, axis=1)

# Actual class labels
true_classes = eval_data.classes

# Disease names
class_labels = list(eval_data.class_indices.keys())

# Generate classification report
report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels,
    output_dict=True
)

# Display heading
print("\n===================================================================================================")
print("                         RESNET50 DISEASE-WISE EVALUATION METRICS (%)")
print("===================================================================================================\n")

# Table header
print(
    f"{'Disease':30} "
    f"{'Accuracy':12} "
    f"{'Precision':12} "
    f"{'Recall':12} "
    f"{'F1-Score':12} "
    f"{'Performance':15}"
)

print("-" * 110)

# Display metrics for each disease
for disease in class_labels:

    precision = report[disease]['precision'] * 100
    recall = report[disease]['recall'] * 100
    f1 = report[disease]['f1-score'] * 100

    # Using Recall as class-wise accuracy
    accuracy = recall

    # Performance category based on F1-score
    if f1 >= 85:
        performance = "Excellent"

    elif f1 >= 70:
        performance = "Good"

    elif f1 >= 50:
        performance = "Moderate"

    else:
        performance = "Poor"

    # Print results
    print(
        f"{disease:30} "
        f"{accuracy:10.2f}% "
        f"{precision:10.2f}% "
        f"{recall:10.2f}% "
        f"{f1:10.2f}% "
        f"{performance:15}"
    )

# Overall Accuracy
overall_accuracy = report['accuracy'] * 100

print("\n===================================================================================================")
print(f"Overall Model Accuracy : {overall_accuracy:.2f}%")
print("===================================================================================================")