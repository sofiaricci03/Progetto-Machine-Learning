"""TRAINING TRANSFER LEARNING - ResNet18"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim

from data.DataSets.fer_dataset import create_dataloaders
from pretrained_model import get_pretrained_model


# Carica configurazione da config.json
import json
with open('../configs/config.json', 'r') as f:
    config = json.load(f)

DATA_PATH = config["train_data_dir"]
BATCH_SIZE = config["batch_size"]
LR = config["learning_rate"]
EPOCHS = config["num_epochs"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = config["pretrained_model_save_path"]

# Trasformazioni
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_loader, val_loader, test_loader, classes = create_dataloaders(
    base_path=DATA_PATH,
    batch_size=BATCH_SIZE,
    train_transform=train_transform,
    augment=True
)

# Modello pre-trained
model = get_pretrained_model(num_classes=len(classes)).to(DEVICE)

# Class imbalance
base_path = Path(DATA_PATH)
counts = [len(list((base_path / cls).glob("*.jpg"))) for cls in classes]
total = sum(counts)
weights = torch.tensor([total/c for c in counts], dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0

# Training
for epoch in range(EPOCHS):
    model.train()
    correct = 0
    total_samples = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    train_acc = correct / total_samples

    # VALIDATION
    model.eval()
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total_samples
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✔ Modello migliorato salvato!")
