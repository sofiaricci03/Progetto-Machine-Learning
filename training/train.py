"""Addestramento con rete neurale custom"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from data.DataSets.fer_dataset import create_dataloaders
import os
import json


BASE_DIR = Path(__file__).resolve().parent.parent
from utils.config_validator import load_and_validate_config

CONFIG_PATH = BASE_DIR / "configs" / "config.json"
SCHEMA_PATH = BASE_DIR / "configs" / "config_schema.json"

config, BASE_DIR = load_and_validate_config(CONFIG_PATH, SCHEMA_PATH, base_dir=BASE_DIR, check_paths=True)

DATA_PATH = BASE_DIR / config["dataset"]["paths"]["train_root"]
BATCH_SIZE = config["training"]["batch_size"]
LR = config["training"]["learning_rate"]
EPOCHS = config["training"]["epochs"]
checkpoint = config["training"]["checkpoint"]
MODEL_SAVE_PATH = BASE_DIR / checkpoint["dir"] / checkpoint["best_name"]

# Device selection
dev = config.get("device", "auto")
if dev == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(dev)


# Trasformazioni

from torchvision import transforms

img_size = config["transforms"]["img_size"]
mean = config["transforms"]["normalize"]["mean"]
std = config["transforms"]["normalize"]["std"]

train_tfms = [transforms.Resize((img_size, img_size))] if config["transforms"]["train"]["resize"] else []
if config["transforms"]["train"]["augmentation"] is not None:
    aug = config["transforms"]["train"]["augmentation"]
    train_tfms.append(transforms.RandomResizedCrop(img_size, scale=tuple(aug["random_resized_crop_scale"])))
    if aug.get("horizontal_flip", False):
        train_tfms.append(transforms.RandomHorizontalFlip())
    if aug.get("rotation_deg", 0) > 0:
        train_tfms.append(transforms.RandomRotation(aug.get("rotation_deg", 0)))

train_tfms.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
train_transform = transforms.Compose(train_tfms) 


# DataLoader

train_loader, val_loader, test_loader, classes = create_dataloaders(
    base_path=DATA_PATH,
    batch_size=BATCH_SIZE,
    train_transform=train_transform,
    augment=True
)

# Istanzia il modello CustomCNN
from training.custom_cnn import CustomCNN
model = CustomCNN(num_classes=len(classes)).to(DEVICE)

# Loss con pesi per classi sbilanciate
base_path = Path(DATA_PATH)
counts_list = [len(list((base_path / cls).glob("*.jpg"))) for cls in classes]

if config["training"]["loss"]["class_weights"] == "inverse_frequency":
    total_images = sum(counts_list)
    weights = [total_images / c for c in counts_list]
    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
else:
    criterion = nn.CrossEntropyLoss()

# Ottimizzatore
opt_name = config["training"]["optimizer"].lower()
wd = config["training"].get("weight_decay", 0.0)
if opt_name == "adam":
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=wd)
elif opt_name == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=wd, momentum=0.9)
else:
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=wd)

# Ensure checkpoint dir exists
(MODEL_SAVE_PATH.parent).mkdir(parents=True, exist_ok=True)

# Training loop

best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Salvataggio modello migliore
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Modello salvato con Val Acc: {best_val_acc:.4f}")
