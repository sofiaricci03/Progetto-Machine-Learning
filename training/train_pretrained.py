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

BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR / "configs" / "config.json", "r") as f:
    config = json.load(f)

DATA_PATH = BASE_DIR / config["train_data_dir"]
BATCH_SIZE = config["batch_size"]
LR = config["learning_rate"]
EPOCHS = config["num_epochs"]
MODEL_SAVE_PATH = BASE_DIR / config["pretrained_model_save_path"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    augment=True    #attiva la creazione di immagini variate per rendere il training più robusto
)

# Modello pre-trained ResNet18
model = get_pretrained_model(num_classes=len(classes)).to(DEVICE)

# Class imbalance
base_path = Path(DATA_PATH)
counts = [len(list((base_path / cls).glob("*.jpg"))) for cls in classes]    #crea lista con il numero di immagini per ogni emozione
total = sum(counts)
weights = torch.tensor([total/c for c in counts], dtype=torch.float32).to(DEVICE)   #calcola i pesi per la CrossEntropyLoss

criterion = nn.CrossEntropyLoss(weight=weights) #grazie al parametro weight, la loss darà più importanza alle classi con meno esempi
optimizer = optim.Adam(model.parameters(), lr=LR)   #parametri del modello pre-trained

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
