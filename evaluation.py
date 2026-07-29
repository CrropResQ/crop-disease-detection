import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef
)


def evaluate_model(
    model,
    history,
    val_data,
    model_name,
    training_time=None,
    model_path=None
):


    print("=" * 60)
    print(f"Evaluating {model_name}")
    print("=" * 60)

    output_dir = os.path.join(
        os.getcwd(),
        "evaluation_results",
        model_name
    )

    os.makedirs(output_dir, exist_ok=True)

    # Reset generator
    val_data.reset()

    # Prediction time
    start = time.time()

    predictions = model.predict(val_data, verbose=1)

    prediction_time = time.time() - start

    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = val_data.classes

    class_labels = list(val_data.class_indices.keys())

    # ===========================
    # Metrics
    # ===========================

    accuracy = accuracy_score(true_classes, predicted_classes)

    precision_macro = precision_score(
        true_classes,
        predicted_classes,
        average="macro",
        zero_division=0
    )

    precision_weighted = precision_score(
        true_classes,
        predicted_classes,
        average="weighted",
        zero_division=0
    )

    recall_macro = recall_score(
        true_classes,
        predicted_classes,
        average="macro",
        zero_division=0
    )

    recall_weighted = recall_score(
        true_classes,
        predicted_classes,
        average="weighted",
        zero_division=0
    )

    f1_macro = f1_score(
        true_classes,
        predicted_classes,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        true_classes,
        predicted_classes,
        average="weighted",
        zero_division=0
    )

    kappa = cohen_kappa_score(
        true_classes,
        predicted_classes
    )

    mcc = matthews_corrcoef(
        true_classes,
        predicted_classes
    )

    # ===========================
    # Print Metrics
    # ===========================

    print("\n========== Overall Performance ==========\n")

    print(f"Accuracy             : {accuracy*100:.2f}%")
    print(f"Precision (Macro)    : {precision_macro:.4f}")
    print(f"Precision (Weighted) : {precision_weighted:.4f}")

    print(f"Recall (Macro)       : {recall_macro:.4f}")
    print(f"Recall (Weighted)    : {recall_weighted:.4f}")

    print(f"F1 Score (Macro)     : {f1_macro:.4f}")
    print(f"F1 Score (Weighted)  : {f1_weighted:.4f}")

    print(f"Cohen Kappa          : {kappa:.4f}")
    print(f"Matthews CC          : {mcc:.4f}")

    if training_time is not None:
        print(f"Training Time (sec)  : {training_time:.2f}")

    print(f"Prediction Time(sec) : {prediction_time:.2f}")

    # ===========================
    # Save Metrics
    # ===========================

    metrics = {
        "Accuracy": accuracy,
        "Precision_Macro": precision_macro,
        "Precision_Weighted": precision_weighted,
        "Recall_Macro": recall_macro,
        "Recall_Weighted": recall_weighted,
        "F1_Macro": f1_macro,
        "F1_Weighted": f1_weighted,
        "Cohen_Kappa": kappa,
        "Matthews_CC": mcc,
        "Prediction_Time_sec": prediction_time
    }

    if training_time is not None:
        metrics["Training_Time_sec"] = training_time

    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_dir, "metrics.csv"),
        index=False
    )

    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:

        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    # ===========================
    # Classification Report
    # ===========================

    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_labels
    )

    print(report)

    with open(
        os.path.join(output_dir, "classification_report.txt"),
        "w"
    ) as f:

        f.write(report)

    # ===========================
    # Predictions CSV
    # ===========================

    predictions_df = pd.DataFrame({
        "Actual": true_classes,
        "Predicted": predicted_classes
    })

    predictions_df.to_csv(
        os.path.join(output_dir, "predictions.csv"),
        index=False
    )

    # ===========================
    # Confusion Matrix
    # ===========================

    cm = confusion_matrix(
        true_classes,
        predicted_classes
    )

    plt.figure(figsize=(10,8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title(f"{model_name} Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "confusion_matrix.png"
        )
    )

    plt.close()

    # ===========================
    # Accuracy Graph
    # ===========================

    plt.figure(figsize=(8,6))

    plt.plot(history.history["accuracy"])

    plt.plot(history.history["val_accuracy"])

    plt.title(f"{model_name} Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend(["Train","Validation"])

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "accuracy.png"
        )
    )

    plt.close()

    # ===========================
    # Loss Graph
    # ===========================

    plt.figure(figsize=(8,6))

    plt.plot(history.history["loss"])

    plt.plot(history.history["val_loss"])

    plt.title(f"{model_name} Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend(["Train","Validation"])

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "loss.png"
        )
    )

    plt.close()

    # ===========================
    # Model Information
    # ===========================

    # model_size = os.path.getsize(model_name + ".keras") / (1024 * 1024)
if model_path is not None and os.path.exists(model_path):
    model_size = os.path.getsize(model_path) / (1024 * 1024)
else:
    model_size = 0




    total_params = model.count_params()

    trainable_params = np.sum(
        [np.prod(v.shape) for v in model.trainable_weights]
    )

    non_trainable_params = np.sum(
        [np.prod(v.shape) for v in model.non_trainable_weights]
    )

    with open(
        os.path.join(output_dir, "model_summary.txt"),
        "w"
    ) as f:

        f.write(f"Model : {model_name}\n")
        f.write(f"Model Size : {model_size:.2f} MB\n")
        f.write(f"Total Parameters : {total_params}\n")
        f.write(f"Trainable Parameters : {trainable_params}\n")
        f.write(f"Non Trainable Parameters : {non_trainable_params}\n")

    print("\nEvaluation Completed Successfully.")