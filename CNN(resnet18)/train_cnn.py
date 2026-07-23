import torch
from torchvision import datasets, transforms, models

# -------- IMAGE TRANSFORM --------
transform = transforms.Compose([   
    transforms.Resize((224,224)),
    transforms.ToTensor() 
])

# -------- LOAD DATASET --------
train_data = datasets.ImageFolder(
    'data',
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=8,
    shuffle=True
)

print(f"Loaded {len(train_data)} images across {len(train_data.classes)} classes")

# -------- LOAD CNN MODEL --------
model = models.resnet18(pretrained=True)

# Replace final layer
model.fc = torch.nn.Linear(
    model.fc.in_features,
    len(train_data.classes)
)

# -------- OPTIMIZER --------
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# -------- TRAINING --------
for epoch in range(1):

    print(f"Epoch {epoch} starting...")

    for images, labels in train_loader:

        outputs = model(images)

        loss = torch.nn.functional.cross_entropy(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch} done")

# -------- SAVE MODEL --------
torch.save(
    model.state_dict(),
    "models/cnn.pth"
)

print("CNN model saved!")
