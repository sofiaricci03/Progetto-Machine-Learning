"""Addestramento con rete neurale custom"""
import torch
import torch.nn as nn
import torch.optim as optim
from data.DataSets.fer_dataset import create_dataloaders
import os
from pathlib import Path
import json


# Carica configurazione da config.json
with open('configs/config.json', 'r') as f:
    config = json.load(f)   #trasforma il contenuto del file JSON in un dizionario Python chiamato config, così da accedere ai valori usando le chiavi

DATA_PATH = config["train_data_dir"]
BATCH_SIZE = config["batch_size"]
LR = config["learning_rate"]
EPOCHS = config["num_epochs"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #usa la GPU se disponibile, altrimenti la CPU
MODEL_SAVE_PATH = config["model_save_path"]

# Trasformazioni

from torchvision import transforms

train_transform = transforms.Compose([  #applica una serie di trasformazioni alle immagini di addestramento
    transforms.Resize((48,48)), #stessa gandezza per tutte le immagini
    transforms.ToTensor(),  #conveerte immagini in matrici
    transforms.Normalize(mean=[0.5], std=[0.5]) #sottrae 0,5 e divide per 0.5 i pixel in modo che rientrino nel range -1 e 1
])


#Creazione DataLoader
#Passa il percorso dei dati, la dimensione del batch e le trasformazioni

train_loader, val_loader, test_loader, classes = create_dataloaders(
    base_path=DATA_PATH,
    batch_size=BATCH_SIZE,
    train_transform=train_transform,
    augment=True #applica trasformazioni di data augmentation durante l'addestramento
)

# CNN custom semplice
# definisce la struttura della rete neurale, gestisce il problema delle classi sbilanciate e imposta l'ottimizzatore
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(      #nn.Sequential consente di concatenare più layer in modo sequenziale, così da evitare di dover definire il metodo forward manualmente per ogni layer
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),       #trasforma le matrici 2D in un vettore di 64 * 12 * 12 elementi
            nn.Linear(64*12*12, 128),   #primo livello connesso con 128 neuroni
            nn.ReLU(),
            nn.Dropout(0.5),    #evita overfitting disattivando casualmente il 50% dei neuroni durante l'addestramento
            nn.Linear(128, num_classes) #livello di output con un neurone per ogni classe
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=len(classes)).to(DEVICE) #sposta il modello sulla GPU se disponibile

# Loss con pesi per classi sbilanciate
#trasforma una stringa in un oggetto Path per gestire i percorsi dei file
base_path = Path(DATA_PATH)
counts_list = [
    len(list((base_path / cls).glob("*.jpg"))) #conta quante immagini ci sono nella cartella di ogni classe
    for cls in classes]
total_images = sum(counts_list) #somma il totale delle immagini nel db
weights = [total_images/c for c in counts_list] #questo calcolo assegna un peso maggiore alle classi con meno immagini: poche immagini = peso alto 
weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE) #converte la lista di pesi in un tensore PyTorch e lo sposta sulla GPU se disponibile
criterion = nn.CrossEntropyLoss(weight=weights)

# Ottimizzatore
# utilizza Adam, un algoritmo di ottimizzazione che adatta i tassi di apprendimento per ogni parametro
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
#variabile per tenere traccia della migliore accuratezza di validazione
best_val_acc = 0.0

for epoch in range(EPOCHS): # ciclo di addestramento per il numero di epochs specificati
    model.train()   #modalità addestramento
    running_loss = 0.0
    correct = 0
    total = 0   #azzera il contatore totale delle immagini

    for images, labels in train_loader: 
        images, labels = images.to(DEVICE), labels.to(DEVICE)   #sposta i dati sulla GPU se non si spostasse, il computer cercherebbe di calcolare i dati in due posti diversi, causando un errore
        optimizer.zero_grad()   #azzera i gradienti calcolati nel passo precedente
        outputs = model(images) #passa le immagini nel modello e la rete restituisce dei punteggi per ogni classe
        loss = criterion(outputs, labels)   #calcola la loss confrontando i punteggi previsti con le etichette reali per vedere quanto scarto di errore c'è
        loss.backward() #il modello torna indietro nei libìvelli per capire chi ha causato l'errore e calcola i gradienti
        optimizer.step()    #aggiorna i pesi del modello in base ai gradienti calcolati per ridurre l'errore successivamente

#calcolo delle statistiche in realtime
        running_loss += loss.item() * images.size(0)    #somma la loss(l'errore) moltiplicata per il numero di immagini nel batch corrente
        _, predicted = torch.max(outputs, 1)    #ottiene le classi previste selezionando l'indice con il punteggio più alto per ogni immagine
        total += labels.size(0) #aggiorna il contatore totale delle immagini
        correct += (predicted == labels).sum().item()   #confronta le classi previste con le etichette reali e conta quante sono corrette

    train_loss = running_loss / total   #calcola media della loss per l'epoch
    train_acc = correct / total  #calcola accuratezza di addestramento (più è vicina a 1.0, meglio è)

    # Validation
    #il modello vine testato su dati che non ha mai visto prima per valutare le sue prestazioni
    model.eval()    #modalità valutazione, quindi disabilita dropout e batch norm(che servivano in fase addestramento)
    val_correct = 0 #inizializza contatori per accuratezza di validazione a zero
    val_total = 0
    with torch.no_grad():   #disabilita il calcolo dei gradienti per risparmiare memoria e velocizzare i calcoli
        for images, labels in val_loader:   #+
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images) #passa le immagini nel modello per ottenere le previsioni
            _, predicted = torch.max(outputs, 1)    #ottiene le classi previste selezionando l'indice con il punteggio più alto per ogni immagine
            val_total += labels.size(0) #aggiorna il contatore totale delle immagini di validazione
            val_correct += (predicted == labels).sum().item()   #confronta le classi previste con le etichette reali e conta quante sono corrette
   
    val_acc = val_correct / val_total #calcola la percentuale di precisione finale
    
    #mostra come procede l'addestramento
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}") 

    # Salvataggio modello migliore
    if val_acc > best_val_acc:  #valuta se l'accuratezza ottenuta è la migliore fino ad ora
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH) #salva i pesi del modello nel percorso specificato
        print(f"Modello salvato con Val Acc: {best_val_acc:.4f}")

        #anche se alla fine dell'addestramento, ottengo un modello non performante, avrò sempre salvato il migliore durante il processo