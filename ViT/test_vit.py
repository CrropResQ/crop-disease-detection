import torch
from torchvision import datasets, transforms
from transformers import ViTForImageClassification

# -------- IMAGE TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------- LOAD DATASET --------
test_data = datasets.ImageFolder(
    'data',
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=8,
    shuffle=False
)

# -------- LOAD ViT MODEL --------
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(test_data.classes),
    ignore_mismatched_sizes=True
)

model.load_state_dict(
    torch.load("models/vit.pth", map_location="cpu")
)

model.eval()

# -------- ACCURACY --------
correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images).logits

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"ViT Accuracy: {accuracy:.2f}%")
