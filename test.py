import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from data.DataSets.fer_dataset import create_dataloaders
from custom_cnn import CustomCNN   # il tuo modello
from pathlib import Path


# ------------------------------
# CONFIG
# ------------------------------
DATA_PATH = "data/DataSets/train"
MODEL_PATH = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# TRANSFORM (validation/test)
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# ------------------------------
# LOAD TEST SET
# ------------------------------
_, _, test_loader, classes = create_dataloaders(
    base_path=DATA_PATH,
    batch_size=64,
    train_transform=transform,
    val_test_transform=transform,
    augment=False
)


# ------------------------------
# LOAD MODEL
# ------------------------------
model = SimpleCNN(num_classes=len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

criterion = nn.CrossEntropyLoss()


# ------------------------------
# TEST LOOP
# ------------------------------
test_loss = 0
correct = 0
total = 0

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

test_loss /= total
test_acc = correct / total


# ------------------------------
# RESULTS
# ------------------------------
print("\n=== RISULTATI TEST SET ===")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}\n")

print("=== CLASSIFICATION REPORT ===")
print(classification_report(all_labels, all_preds, target_names=classes))

print("=== CONFUSION MATRIX ===")
print(confusion_matrix(all_labels, all_preds))
