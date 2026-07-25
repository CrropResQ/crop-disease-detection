import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from tqdm import tqdm


def main():

    # -------------------------------
    # DEVICE
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # -------------------------------
    # DATASET PATH
    # -------------------------------
    DATASET_PATH = r"C:\Users\Vaishnav\Downloads\archive\Rice_Leaf_Diease\Rice_Leaf_Diease\train"

    # -------------------------------
    # IMAGE TRANSFORMS
    # -------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # -------------------------------
    # LOAD DATASET
    # -------------------------------
    train_data = datasets.ImageFolder(
        DATASET_PATH,
        transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        num_workers=0,      # IMPORTANT FOR WINDOWS
        pin_memory=True
    )

    print(f"\nLoaded {len(train_data)} images")
    print(f"Detected {len(train_data.classes)} classes")
    print(train_data.classes)

    # -------------------------------
    # LOAD PRETRAINED ViT
    # -------------------------------
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(train_data.classes),
        ignore_mismatched_sizes=True
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5
    )

    criterion = torch.nn.CrossEntropyLoss()

    epochs = 5

    # -------------------------------
    # TRAINING
    # -------------------------------
    model.train()

    for epoch in range(epochs):

        running_loss = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}"
        )

        for images, labels in progress:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images).logits

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)

        print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # -------------------------------
    # SAVE MODEL
    # -------------------------------
    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), "models/vit_rice.pth")

    print("\nTraining Complete!")
    print("Model saved to models/vit_rice.pth")


if __name__ == "__main__":
    main()