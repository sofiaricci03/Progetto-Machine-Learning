# Questo file definisce:
#  - Un dataset custom PyTorch per FER2013 (FER2013Dataset)
#  - Una funzione per creare DataLoader per train/val/test
#  - Gestione delle trasformazioni e data augmentation

# Utilizzato per caricare, preprocessare e suddividere le immagini del dataset FER2013 per 
# il training e la valutazione di modelli di riconoscimento delle emozioni da immagini.


from pathlib import Path  # Gestione path cross-platform
from PIL import Image    # Apertura immagini
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


def set_seed(seed=42):
    """
    Imposta il seed per tutti i generatori di numeri casuali per garantire la riproducibilità.
    
    Args:
        seed: valore del seed (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # per multi-GPU
    
    # Per massima riproducibilità (può rallentare leggermente il training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(cfg, mode="train"):
    """
    Costruisce le trasformazioni da applicare al dataset basandosi sul file di configurazione.
    
    Args:
        cfg: dizionario di configurazione caricato da config.json
        mode: "train" per training set (con augmentation), "val" o "test" per validation/test (senza augmentation)
    
    Returns:
        transforms.Compose: pipeline di trasformazioni configurata
    """
    img_size = cfg["transforms"]["img_size"]
    mean = cfg["transforms"]["normalize"]["mean"]
    std = cfg["transforms"]["normalize"]["std"]

    tfms = []
    
    if mode == "train":
        if cfg["transforms"]["train"]["resize"]:
            tfms.append(transforms.Resize((img_size, img_size)))
        
        aug = cfg["transforms"]["train"]["augmentation"]
        if aug is not None:
            tfms.append(transforms.RandomResizedCrop(
                img_size, 
                scale=tuple(aug["random_resized_crop_scale"])
            ))
            if aug.get("horizontal_flip", False):
                tfms.append(transforms.RandomHorizontalFlip())
            if aug.get("rotation_deg", 0) > 0:
                tfms.append(transforms.RandomRotation(aug["rotation_deg"]))
    else:  # val o test
        if cfg["transforms"]["val_test"]["resize"]:
            tfms.append(transforms.Resize((img_size, img_size)))

    tfms.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(tfms)


# Classe custom Dataset per FER2013
class FER2013Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Dataset PyTorch custom per FER2013.
        Args:
            image_paths: lista di Path alle immagini
            labels: lista di label corrispondenti
            transform: pipeline completa di trasformazioni (include già augmentation se necessaria)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Restituisce il numero totale di immagini nel dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Carica l'immagine e la label corrispondente
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('L')  # converte l'immagine in grayscale

        # Applica tutte le trasformazioni (inclusa augmentation se presente nella pipeline)
        if self.transform:
            img = self.transform(img)

        return img, label



# Funzione per creare DataLoader per train, val e test.
# Il codice guarda dentro la cartella base_path, legge i nomi delle sottocartelle e assegna a ognuna un numero

def create_dataloaders(base_path, batch_size=64, config=None, train_transform=None, val_transform=None, test_transform=None, val_ratio=0.15, test_ratio=0.15, shuffle=True, random_seed=42):
    """
    Crea train, validation e test DataLoader a partire da una struttura di cartelle.
    
    Args:
        base_path: Path della cartella train di FER2013
        batch_size: batch size
        config: dizionario di configurazione (opzionale, usato per costruire trasformazioni automaticamente)
        train_transform: trasformazioni da applicare al train (se None e config è fornito, usa build_transforms)
        val_transform: trasformazioni da applicare alla validation (se None e config è fornito, usa build_transforms)
        test_transform: trasformazioni da applicare al test (se None e config è fornito, usa build_transforms)
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

    # Usa il seed dal config se disponibile, altrimenti usa il parametro random_seed
    if config is not None:
        random_seed = config.get("seed", random_seed)
    
    # Utilizza train_test_split con stratify per garantire che la distribuzione delle classi sia bilanciata in tutti i set (train/val/test)
    # stratify=all_labels assicura che ogni split mantenga le stesse proporzioni delle classi
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        all_images, all_labels, 
        test_size=(val_ratio + test_ratio), 
        stratify=all_labels, 
        random_state=random_seed
    )
    
    # Divide temp in validation e test mantenendo le proporzioni
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, 
        test_size=(test_ratio / (val_ratio + test_ratio)),
        stratify=temp_labels, 
        random_state=random_seed
    )

    # Costruisce le trasformazioni dal config se non fornite esplicitamente
    if config is not None:
        if train_transform is None:
            train_transform = build_transforms(config, mode="train")
        if val_transform is None:
            val_transform = build_transforms(config, mode="val")
        if test_transform is None:
            test_transform = build_transforms(config, mode="test")
    else:
        # Fallback se non c'è config
        val_transform = val_transform if val_transform is not None else train_transform
        test_transform = test_transform if test_transform is not None else train_transform

    train_dataset = FER2013Dataset(train_imgs, train_labels, transform=train_transform)
    val_dataset = FER2013Dataset(val_imgs, val_labels, transform=val_transform)
    test_dataset = FER2013Dataset(test_imgs, test_labels, transform=test_transform)

    # Crea i DataLoader per batch processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, classes
