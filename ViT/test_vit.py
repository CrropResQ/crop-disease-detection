import os
import torch
from torchvision import datasets, transforms
from transformers import ViTForImageClassification

# -------------------------------
# DEVICE
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# -------------------------------
# PATHS
# -------------------------------
TRAIN_PATH = r"C:\Users\Vaishnav\Downloads\archive\Rice_Leaf_Diease\Rice_Leaf_Diease\train"
TEST_PATH  = r"C:\Users\Vaishnav\Downloads\archive\Rice_Leaf_Diease\Rice_Leaf_Diease\test"

# -------------------------------
# TRANSFORMS
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------------------------------
# LOAD TRAIN DATASET
# (Only to get the class mapping)
# -------------------------------
train_data = datasets.ImageFolder(
    TRAIN_PATH,
    transform=transform
)

print("\nTraining Classes")
print(train_data.classes)

# -------------------------------
# LOAD TEST DATASET
# -------------------------------
test_data = datasets.ImageFolder(
    TEST_PATH,
    transform=transform
)

print("\nOriginal Test Classes")
print(test_data.classes)

# Force the test dataset to use the
# same mapping as the training dataset

test_data.class_to_idx = train_data.class_to_idx

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(train_data.classes),
    ignore_mismatched_sizes=True
)

model.load_state_dict(
    torch.load("models/vit_rice.pth", map_location=device)
)

model.to(device)
model.eval()

# -------------------------------
# TEST
# -------------------------------
correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images).logits

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"\nTest Accuracy : {accuracy:.2f}%")