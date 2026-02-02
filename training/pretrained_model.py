import torch
import torch.nn as nn
from torchvision import models

class PretrainedResNet(nn.Module):  #definisce un modello ResNet pre-addestrato con adattamento per FER2013
    def __init__(self, num_classes):
        super().__init__()

        # Carica ResNet-18 pre-addestrata. Questo modello sa già riconoscere forme, texture e contorni
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Adatta il primo layer convoluzionale perchè Res-Net è stato addestrato su immagini a colori (3 canali)
        # FER2013 è in scala di grigi → sotituiamo il primo layer convoluzionale per accettare immagini a singolo canale
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False   #parametri invariati
        )

        # Blocca tutti i layer in mod da non cambiare i pesi durante l'addestramento
        for param in self.model.parameters():
            param.requires_grad = False

        # Sblocca l’ultimo blocco. Gli ultimi layer riconoscono le forme complesse 
        #Permettiamo al modello di specializzarsi sulle caratteristiche specifiche delle espressioni facciali 
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Cambia il classificatore finale
        in_features = self.model.fc.in_features #recupera il numero di neuroni in ingresso all'ultimo layer
        # Sostituisce l'ultimo layer con un nuovo livello lineare che punta alle tue 7 emozioni
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Funzione di utilità per creare il modello pre-addestrato
def get_pretrained_model(num_classes):
    return PretrainedResNet(num_classes)