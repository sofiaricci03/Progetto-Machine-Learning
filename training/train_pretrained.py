"""TRAINING TRANSFER LEARNING - ResNet18"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim

from data.DataSets.fer_dataset import create_dataloaders
from pretrained_model import get_pretrained_model


# Carica e valida configurazione
from utils.config_validator import load_and_validate_config

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "config.json"
SCHEMA_PATH = BASE_DIR / "configs" / "config_schema.json"

config, BASE_DIR = load_and_validate_config(CONFIG_PATH, SCHEMA_PATH, base_dir=BASE_DIR, check_paths=True)

DATA_PATH = BASE_DIR / config["dataset"]["paths"]["train_root"]
BATCH_SIZE = config["training"]["batch_size"]
LR = config["training"]["learning_rate"] if "learning_rate" in config["training"] else config["training"].get("learning_rate", 0.001)
EPOCHS = config["training"]["epochs"]
checkpoint = config["training"]["checkpoint"]
MODEL_SAVE_PATH = BASE_DIR / checkpoint["dir"] / checkpoint["best_name"]

# Device selection (auto/cpu/cuda)
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

train_tfms = []
if config["transforms"]["train"]["resize"]:
    train_tfms.append(transforms.Resize((img_size, img_size)))

aug = config["transforms"]["train"]["augmentation"]
if aug is not None:
    train_tfms.append(transforms.RandomResizedCrop(img_size, scale=tuple(aug["random_resized_crop_scale"])))
    if aug.get("horizontal_flip", False):
        train_tfms.append(transforms.RandomHorizontalFlip())
    if aug.get("rotation_deg", 0) > 0:
        train_tfms.append(transforms.RandomRotation(aug.get("rotation_deg", 0)))

train_tfms.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

train_transform = transforms.Compose(train_tfms)

train_loader, val_loader, test_loader, classes = create_dataloaders(
    base_path=DATA_PATH,
    batch_size=BATCH_SIZE,
    train_transform=train_transform,
    augment=True    #attiva la creazione di immagini variate per rendere il training più robusto
)

# Modello pre-trained ResNet18
model = get_pretrained_model(num_classes=len(classes)).to(DEVICE)

# Class imbalance / criterion
base_path = Path(DATA_PATH)
counts = [len(list((base_path / cls).glob("*.jpg"))) for cls in classes]    #crea lista con il numero di immagini per ogni emozione

if config["training"]["loss"]["class_weights"] == "inverse_frequency":
    total = sum(counts)
    weights = torch.tensor([total/c for c in counts], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
else:
    criterion = nn.CrossEntropyLoss()

# Optimizer
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

best_val_acc = 0    #inizializza la migliore accuratezza di validazione

# Training
for epoch in range(EPOCHS): #ripete il ciclo per il numero di epoche specificato
    model.train()
    correct = 0 #inizializza i contatori
    total_samples = 0
    
    #Il train_loader fornisce le immagini a piccoli gruppi (batch). Spostiamo tutto su DEVICE (GPU o CPU) per poter fare i calcoli
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()   #pulisce da errori precedenti
        outputs = model(images) #il modello fa una previsione
        loss = criterion(outputs, labels)   #calcola quanto è grande l'errore commesso 
        loss.backward() #calcola come muoveris per ridurre l'errore
        optimizer.step()    #applica correzioni ai pesi del modello
        
        _, predicted = outputs.max(1)   #prende l'emozione con la probabilità più alta
        correct += (predicted == labels).sum().item()   #somma tutte le volte che la previsione è uguale alla realtà
        total_samples += labels.size(0) #conta il numero totale di immagini processate

    train_acc = correct / total_samples #calcola percentuale di risposte corrette

    # VALIDATION
    #entraimo in modalità valutazione quindi si azzerano i contatori e non si calcolano i gradienti
    model.eval()
    correct = 0
    total_samples = 0
    with torch.no_grad():
        #si prende un set di immagini e si spostano su DEVICE per fare i calcoli
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)   #si fanno previsioni e prende la classe con probabilità più alta 
            total_samples += labels.size(0) #conta le previsioni esatte fatte
            correct += (predicted == labels).sum().item()

    val_acc = correct / total_samples
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")


#Se l'accuratezza di validazione è migliore di quella precedente, salva i pesi del modello per recuperarli in futuro
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✔ Modello migliorato salvato!")
