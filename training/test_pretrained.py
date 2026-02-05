"""TEST PRETRAINED MODEL"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from sklearn.metrics import classification_report, confusion_matrix
from pretrained_model import get_pretrained_model
from data.DataSets.fer_dataset import create_dataloaders


# Carica e valida configurazione
from utils.config_validator import load_and_validate_config

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "config.json"
SCHEMA_PATH = BASE_DIR / "configs" / "config_schema.json"

config, BASE_DIR = load_and_validate_config(CONFIG_PATH, SCHEMA_PATH, base_dir=BASE_DIR, check_paths=True)

DATA_PATH = BASE_DIR / config["dataset"]["paths"]["test_root"]
checkpoint = config["training"]["checkpoint"]
MODEL_PATH = BASE_DIR / checkpoint["dir"] / checkpoint["best_name"]

# Device selection
dev = config.get("device", "auto")
if dev == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(dev) 


# Trasformazione
from torchvision import transforms

img_size = config["transforms"]["img_size"]
mean = config["transforms"]["normalize"]["mean"]
std = config["transforms"]["normalize"]["std"]

test_tfms = []
if config["transforms"]["val_test"]["resize"]:
    test_tfms.append(transforms.Resize((img_size, img_size)))

test_tfms.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

test_transform = transforms.Compose(test_tfms)

# DataLoader
_, _, test_loader, classes = create_dataloaders(
    base_path=DATA_PATH,
    batch_size=config["training"]["batch_size"],
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
