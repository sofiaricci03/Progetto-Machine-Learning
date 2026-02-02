#È una rete neurale convoluzionale che imparerà a riconoscere le emozioni dalle immagini 48x48 in scala di grigi

import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module): #definisce una rete neurale convoluzionale personalizzata
    def __init__(self, num_classes=7): #7 classi di emozioni
        super(CustomCNN, self).__init__() #inizializza la classe base nn.Module

#definisce i layer della rete
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) #prende 1 canale (grigio) e ne crea 32, usando una finestra 3x3. Il padding=1 serve a non far rimpicciolire l'immagine
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    #prende i 32 canali precedenti e ne crea 64, cercando pattern più complessi
        self.pool = nn.MaxPool2d(2, 2)#riduce la dimesione delle immagini da 48 a 24 e poi a 12 in modo da ridurre il numero di parametri e calcoli

        # dopo due pool 48x48 -> 24x24 -> 12x12
        self.fc1 = nn.Linear(64 * 12 * 12, 128) #layer completamente connesso che prende i 64 canali di 12x12 pixel e li riduce a 128 neuroni
        self.fc2 = nn.Linear(128, num_classes) #layer di output che mappa i 128 neuroni alle 7 classi di emozioni

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    #applica la convoluzione1, la funzione di attivazione ReLU e il pooling. La Relu mi servirà per decidere quali valori anallizzare e quali no
        x = self.pool(F.relu(self.conv2(x)))    #applica la convoluzione2 , quindi l'immagine diventa una mappa da 12x12 pixel
        x = x.view(x.size(0), -1)              #appiattisce la mappa in un vettore per poterlo passare ai layer completamente connessi
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
