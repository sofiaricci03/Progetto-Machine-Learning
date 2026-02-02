"""TEST PRETRAINED MODEL"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from sklearn.metrics import classification_report, confusion_matrix
from pretrained_model import get_pretrained_model
from data.DataSets.fer_dataset import create_dataloaders


# Carica configurazione da config.json
import json
BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = BASE_DIR / "configs" / "config.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DATA_PATH = BASE_DIR / config["test_data_dir"]
MODEL_PATH = BASE_DIR / config["pretrained_model_save_path"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Trasformazione
from torchvision import transforms

test_transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# DataLoader
_, _, test_loader, classes = create_dataloaders(
    base_path=DATA_PATH,
    batch_size=config["batch_size"],
    train_transform=test_transform,
    augment=False
)

# Modello
model = get_pretrained_model(num_classes=len(classes)).to(DEVICE)

# Carica i pesi se il file esiste
if Path(MODEL_PATH).exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✓ Modello caricato da {MODEL_PATH}")
else:
    print(f"⚠ File {MODEL_PATH} non trovato. Usando modello con pesi iniziali.")

model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\n📌 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=classes))

print("\n📌 Confusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
