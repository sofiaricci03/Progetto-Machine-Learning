import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import json

from data.DataSets.fer_dataset import create_dataloaders


# CNN custom semplice (must match the one used in training)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*12*12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# CONFIG
BASE_DIR = Path(__file__).resolve().parent.parent
from utils.config_validator import load_and_validate_config

CONFIG_PATH = BASE_DIR / "configs" / "config.json"
SCHEMA_PATH = BASE_DIR / "configs" / "config_schema.json"

config, BASE_DIR = load_and_validate_config(CONFIG_PATH, SCHEMA_PATH, base_dir=BASE_DIR, check_paths=True)

DATA_PATH = BASE_DIR / config["dataset"]["paths"]["test_root"]
checkpoint = config["training"]["checkpoint"]
MODEL_PATH = BASE_DIR / checkpoint["dir"] / checkpoint["best_name"]

# Device
dev = config.get("device", "auto")
if dev == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(dev)


# TRANSFORM (validation/test)
#crea una lista di operazioni da eseguire sulle immagini
img_size = config["transforms"]["img_size"]
mean = config["transforms"]["normalize"]["mean"]
std = config["transforms"]["normalize"]["std"]

transform_list = []
if config["transforms"]["val_test"]["resize"]:
    transform_list.append(transforms.Resize((img_size, img_size)))

transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
transform = transforms.Compose(transform_list)


# LOAD TEST SET
_, _, test_loader, classes = create_dataloaders(    #usiamo gli underscore per ignorare i primi due output (train e val loader) perchè serve solo il test_loader
    batch_size=config["training"]["batch_size"],
    train_transform=transform,
    augment=False   #non vogliamo fare data augmentation sul test set
)


# LOAD MODEL
model = SimpleCNN(num_classes=len(classes)) #crea un'istanza del modello SimpleCNN con il numero di classi corretto
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  #carica i pesi salvati del modello dal percorso specificato
model.to(DEVICE)
model.eval()    #disattiva il dropout 

criterion = nn.CrossEntropyLoss()


# TEST LOOP
test_loss = 0   #inizializza la variabile per tenere traccia della perdita totale sul test set
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

        all_labels.extend(labels.cpu().numpy()) #sposta le etichette su cpu e le converte in numpy array
        all_preds.extend(predicted.cpu().numpy()) #fa lo stesso con le predizioni

test_loss /= total
test_acc = correct / total



# RESULTS
#Genera una tabella con tre metriche chiave per ogni emozione
#precisione : quanet volte volte il modello ha fatto una previsione corretta per una classe specifica
#recall: quante volte il modello ha fatto una previsione corretta per una classe specifica
#f1-score: media tra precisione e recall

print("\n=== RISULTATI TEST SET ===")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}\n")

print("=== CLASSIFICATION REPORT ===")
print(classification_report(all_labels, all_preds, target_names=classes))


#tabella che mostra i punti in cui il modello ha confuso delle emozioni con altre
print("=== CONFUSION MATRIX ===")
print(confusion_matrix(all_labels, all_preds))
