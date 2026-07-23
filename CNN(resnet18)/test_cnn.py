import torch
from torchvision import datasets, transforms, models

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

# -------- LOAD MODEL --------
model = models.resnet18(pretrained=False)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    len(test_data.classes)
)

model.load_state_dict(
    torch.load("models/cnn.pth", map_location="cpu")
)

model.eval()

# -------- ACCURACY --------
correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Accuracy: {accuracy:.2f}%")
