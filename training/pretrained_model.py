import torch
import torch.nn as nn
from torchvision import models

class PretrainedResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Carica ResNet-18 pre-addestrata
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # FER2013 è in scala di grigi → convertiamo conv1 da 3 canali a 1
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Blocca tutti i layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Sblocca l’ultimo blocco
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Cambiamo il classificatore finale
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ⚠️ AGGIUNGI QUESTA FUNZIONE QUI SOTTO
def get_pretrained_model(num_classes):
    return PretrainedResNet(num_classes)
