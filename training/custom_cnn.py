"""Rete neurale convoluzionale per il riconoscimento di emozioni da immagini 48x48 in scala di grigi"""

import torch.nn as nn

class CustomCNN(nn.Module): #definisce una rete neurale convoluzionale personalizzata
    """CNN personalizzata per la classificazione delle emozioni (7 classi)"""
    
    def __init__(self, num_classes=7):  #7 classi di emozioni
        super(CustomCNN, self).__init__() #inizializza la classe base nn.Module

        #definisce i layer della rete
        self.features = nn.Sequential(  
            # Primo blocco convoluzionale: 1 -> 32 canali
            nn.Conv2d(1, 32, 3, padding=1), #prende 1 canale (grigio) e ne crea 32, usando una finestra 3x3. Il padding=1 serve a non far rimpicciolire l'immagine
            nn.ReLU(), #funzione di attivazione ReLU per decidere quali valori analizzare e quali no
            nn.MaxPool2d(2), #riduce la dimensione delle immagini da 48 a 24 in modo da ridurre il numero di parametri e calcoli (48x48 -> 24x24)
            
            # Secondo blocco convoluzionale: 32 -> 64 canali
            nn.Conv2d(32, 64, 3, padding=1), #prende i 32 canali precedenti e ne crea 64, cercando pattern più complessi (24x24x32 -> 24x24x64)
            nn.ReLU(), #funzione di attivazione ReLU
            nn.MaxPool2d(2) #riduce la dimensione da 24 a 12, quindi l'immagine diventa una mappa da 12x12 pixel (24x24 -> 12x12)
        )
        
        # Blocco di classificazione - dopo due pool 48x48 -> 24x24 -> 12x12
        self.classifier = nn.Sequential(
            nn.Flatten(), #appiattisce la mappa in un vettore per poterlo passare ai layer completamente connessi
            nn.Linear(64 * 12 * 12, 128), #layer completamente connesso che prende i 64 canali di 12x12 pixel e li riduce a 128 neuroni
            nn.ReLU(),
            nn.Dropout(0.5), #dropout per prevenire overfitting
            nn.Linear(128, num_classes) #layer di output che mappa i 128 neuroni alle 7 classi di emozioni
        )

    def forward(self, x):
        x = self.features(x) #applica i blocchi convoluzionali
        x = self.classifier(x) #applica il blocco di classificazione
        return x