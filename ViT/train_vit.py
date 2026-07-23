import torch
from torchvision import datasets, transforms
from transformers import ViTForImageClassification

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------- LOAD DATA --------
train_data = datasets.ImageFolder('data', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)

print(f"Loaded {len(train_data)} images across {len(train_data.classes)} classes")

# -------- MODEL --------
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(train_data.classes),
    ignore_mismatched_sizes=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# -------- TRAIN --------
for epoch in range(1):   # keep 1 for speed
    print(f"Epoch {epoch} starting...")

    for images, labels in train_loader:
        outputs = model(images).logits
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} done")

# -------- SAVE --------
torch.save(model.state_dict(), "models/vit.pth")
print("ViT model saved!")
