import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt

# ---------------- DEVICE ----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- DATASET PATH ----------------

TEST_PATH = r"C:\Users\Vaishnav\Downloads\archive\Rice_Leaf_Diease\Rice_Leaf_Diease\test"

# ---------------- TRANSFORM ----------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- LOAD TEST DATA ----------------

test_data = datasets.ImageFolder(TEST_PATH, transform=transform)

test_loader = DataLoader(
    test_data,
    batch_size=16,
    shuffle=False
)

print(f"Loaded {len(test_data)} test images")
print("Classes:", test_data.classes)

# ---------------- LOAD MODEL ----------------

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(test_data.classes),
    ignore_mismatched_sizes=True
)

# CHANGE THIS IF YOUR FILE IS vit.pth
model.load_state_dict(
    torch.load("models/vit_rice.pth", map_location=device)
)

model.to(device)
model.eval()

# ---------------- TEST ----------------

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(pixel_values=images)

        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ---------------- ACCURACY ----------------

accuracy = accuracy_score(all_labels, all_preds)

print("\n===================================")
print(f"Test Accuracy : {accuracy * 100:.2f}%")
print("===================================\n")

# ---------------- CLASSIFICATION REPORT ----------------

print(classification_report(
    all_labels,
    all_preds,
    target_names=test_data.classes
))

# ---------------- CONFUSION MATRIX ----------------

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = range(len(test_data.classes))

plt.xticks(
    tick_marks,
    test_data.classes,
    rotation=90
)

plt.yticks(
    tick_marks,
    test_data.classes
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion Matrix saved as confusion_matrix.png")