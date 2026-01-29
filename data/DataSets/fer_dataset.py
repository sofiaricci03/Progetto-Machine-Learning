# Questo file definisce:
#  - Un dataset custom PyTorch per FER2013 (FER2013Dataset)
#  - Una funzione per creare DataLoader per train/val/test
#  - Gestione delle trasformazioni e data augmentation

# Utilizzato per caricare, preprocessare e suddividere le immagini del dataset FER2013 per 
# il training e la valutazione di modelli di riconoscimento delle emozioni da immagini.


from pathlib import Path  # Gestione path cross-platform
from PIL import Image    # Apertura immagini
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random


# Classe custom Dataset per FER2013
class FER2013Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, augment=False):
        """
        Dataset PyTorch custom per FER2013.
        Args:
            image_paths: lista di Path alle immagini
            labels: lista di label corrispondenti
            transform: trasformazioni comuni (resize, tensor, normalize)
            augment: True per applicare augmentation (solo sul train)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment

        # Definizione delle trasformazioni di data augmentation leggere
        
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),                  # flip orizzontale: per riconoscere l'emozione sia se il volto è divolto a dx o sx
            transforms.RandomRotation(15),                      # rotazione: insegna al modello a riconoscere l'emozione anche se capo è inclinato
            transforms.RandomResizedCrop(48, scale=(0.8,1.0))   #crop casuale: simula uno zoom o un leggero cambiamento di inquadratura, costringendo il modello a focalizzarsi sui dettagli anche se non sono esattamente al centro
        ])

    def __len__(self):
        # Restituisce il numero totale di immagini nel dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Carica l'immagine e la label corrispondente
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('L')  # converte l'immagine in grayscale

        # Applica augmentation solo se richiesto (se self.augment è True)
        if self.augment and self.augment_transform:
            img = self.augment_transform(img)
        # Applica le trasformazioni comuni ovvero trasforma l'immagine in una matrice (resize, tensor, normalize)
        if self.transform:
            img = self.transform(img)

        return img, label



# Funzione per creare DataLoader per train, val e test.
# Il codice guarda dentro la cartella base_path, legge i nomi delle sottocartelle e assegna a ognuna un numero

def create_dataloaders(base_path, batch_size=64, train_transform=None, augment=True, val_ratio=0.15, test_ratio=0.15, shuffle=True, random_seed=42):
    """
    Crea train, validation e test DataLoader a partire da una struttura di cartelle:
    base_path/
        classe1/
            img1.jpg
            ...
        classe2/
            ...
    Args:
        base_path: Path della cartella train di FER2013
        batch_size: batch size
        train_transform: trasformazioni da applicare
        augment: True per augmentation sul train
        val_ratio: percentuale validation
        test_ratio: percentuale test
        shuffle: se mischiare i dati prima dello split
        random_seed: seed per la riproducibilità
    Returns:
        train_loader, val_loader, test_loader, classes (lista nomi classi)
    """
    base_path = Path(base_path)
    # Estrae le classi dalle sottocartelle
    classes = [d.name for d in base_path.iterdir() if d.is_dir()]
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    # Raccoglie tutti i path delle immagini e le relative label
    all_images = []
    all_labels = []
    for cls in classes:
        for img_path in (base_path / cls).glob("*.jpg"):
            all_images.append(img_path)
            all_labels.append(class_to_idx[cls])

    # Per evitare che il modello impari l'ordine delle immagini invece delle emozioni, il codice
    # mescola i dati prima di dividerli in train/val/test
    random.seed(random_seed)
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    all_images[:], all_labels[:] = zip(*combined)

    total = len(all_images)
    val_count = int(total * val_ratio)  # val (simulazione d'esame" usata durante il training per regolare i parametri.
    test_count = int(total * test_ratio)    # test (dati finali, mai utilizzati prima, per valutare il modello addestrato)
    train_count = total - val_count - test_count  # train (dati per esercitarsi)

    train_imgs, train_labels = all_images[:train_count], all_labels[:train_count]
    val_imgs, val_labels = all_images[train_count:train_count+val_count], all_labels[train_count:train_count+val_count]
    test_imgs, test_labels = all_images[train_count+val_count:], all_labels[train_count+val_count:]

    # Crea i dataset PyTorch
    train_dataset = FER2013Dataset(train_imgs, train_labels, transform=train_transform, augment=augment)
    val_dataset = FER2013Dataset(val_imgs, val_labels, transform=train_transform, augment=False)
    test_dataset = FER2013Dataset(test_imgs, test_labels, transform=train_transform, augment=False)

    # Crea i DataLoader per batch processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, classes
