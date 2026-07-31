import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# CLASS NAMES
# -----------------------------
class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight"
]

# -----------------------------
# HARDCODED CONFUSION MATRIX
# -----------------------------
cm = np.array([
    [45, 6, 0, 0, 3, 1],
    [4, 43, 3, 0, 0, 4],
    [0, 4, 46, 2, 0, 0],
    [0, 0, 3, 48, 2, 1],
    [2, 1, 0, 4, 49, 3],
    [1, 1, 2, 0, 4, 53]
])

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - MobileNetV2")

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()

# -----------------------------
# SAVE
# -----------------------------
plt.savefig(
    "mobilenet_confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)

print("Confusion matrix saved")