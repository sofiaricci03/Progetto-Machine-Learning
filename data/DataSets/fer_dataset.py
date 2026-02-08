"""
fer_dataset.py

Questo modulo contiene:
- un Dataset custom PyTorch per FER2013 (FER2013Dataset)
- funzioni per:
  - impostare il seed (riproducibilità)
  - costruire le trasformazioni da config.json (preprocessing + augmentation)
  - creare DataLoader per train/val/test con split stratificato

Viene utilizzato per caricare e preprocessare il dataset FER2013 in formato "cartelle per classe",
e per fornire ai modelli (CNN custom / ResNet18) batch di dati coerenti e riproducibili.
"""

from pathlib import Path                 # Gestione path in modo cross-platform (Windows/Linux/Mac)
from PIL import Image                   # Apertura e lettura delle immagini
import random                           # Random per shuffle/split riproducibili
import numpy as np                      # Seed e utilità numeriche
import torch                            # PyTorch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms      # Trasformazioni e data augmentation
from sklearn.model_selection import train_test_split  # Split stratificato train/val/test


# ---------------------------------------------------------------------------
# Riproducibilità
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Imposta il seed per rendere riproducibili le operazioni casuali.
    In particolare:
    - random (python)
    - numpy
    - torch (CPU)
    - torch (GPU)

    Nota: impostare cudnn.deterministic=True e benchmark=False aumenta la riproducibilità,
    ma può rallentare leggermente le performance su GPU.

    Args:
        seed: intero che rappresenta il seed (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Seed per CUDA (se disponibile)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Impostazioni per comportamento deterministico su GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Costruzione trasformazioni da configurazione
# ---------------------------------------------------------------------------

def build_transforms(cfg: dict, mode: str = "train") -> transforms.Compose:
    """
    Costruisce la pipeline di trasformazioni a partire dal config.json.

    - In modalità 'train' applica (se configurata) anche la data augmentation.
    - In modalità 'val' / 'test' NON applica augmentation (solo preprocessing).

    Args:
        cfg: dizionario di configurazione (caricato da config.json)
        mode: "train", "val" oppure "test"

    Returns:
        transforms.Compose: pipeline di trasformazioni pronta per PyTorch Dataset
    """
    img_size = cfg["transforms"]["img_size"]
    mean = cfg["transforms"]["normalize"]["mean"]
    std = cfg["transforms"]["normalize"]["std"]

    tfms = []

    # ---- TRAIN: preprocessing + augmentation (se presente) ----
    if mode == "train":
        if cfg["transforms"]["train"]["resize"]:
            # Resize fisso alla dimensione target
            tfms.append(transforms.Resize((img_size, img_size)))

        aug = cfg["transforms"]["train"]["augmentation"]

        # Se augmentation è attiva, la applichiamo prima di ToTensor/Normalize
        if aug is not None:
            # RandomResizedCrop "simula" zoom e traslazioni mantenendo dimensione finale img_size
            tfms.append(
                transforms.RandomResizedCrop(
                    img_size,
                    scale=tuple(aug["random_resized_crop_scale"])
                )
            )

            # Flip orizzontale (utile per volti)
            if aug.get("horizontal_flip", False):
                tfms.append(transforms.RandomHorizontalFlip())

            # Rotazione casuale (entro un certo angolo)
            if aug.get("rotation_deg", 0) > 0:
                tfms.append(transforms.RandomRotation(aug["rotation_deg"]))

    # ---- VAL/TEST: SOLO preprocessing (NO augmentation) ----
    else:
        if cfg["transforms"]["val_test"]["resize"]:
            tfms.append(transforms.Resize((img_size, img_size)))

    # Conversione a tensore e normalizzazione (sempre)
    tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transforms.Compose(tfms)


# ---------------------------------------------------------------------------
# Dataset custom
# ---------------------------------------------------------------------------

class FER2013Dataset(Dataset):
    """
    Dataset custom PyTorch per FER2013.

    Contiene:
    - lista di path immagini
    - lista label numeriche (0..num_classes-1)
    - trasformazioni (preprocessing + eventualmente augmentation)
    """

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: lista di Path alle immagini
            labels: lista di label intere corrispondenti alle immagini
            transform: transforms.Compose completo (include augmentation se necessario)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Numero totale di campioni nel dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Restituisce il campione idx-esimo.

        Steps:
        1) apre immagine
        2) converte in grayscale (FER2013 è in scala di grigi)
        3) applica trasformazioni (train: con augmentation; val/test: solo preprocessing)
        4) ritorna (immagine_tensor, label)

        Returns:
            img (Tensor), label (int)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Convertiamo sempre a grayscale (1 canale)
        img = Image.open(img_path).convert("L")

        # Applica pipeline trasformazioni (se fornita)
        if self.transform is not None:
            img = self.transform(img)

        return img, label


# ---------------------------------------------------------------------------
# DataLoader: raccolta immagini, split stratificato, creazione dataset/loader
# ---------------------------------------------------------------------------

def create_dataloaders(
    base_path,
    batch_size: int = 64,
    config: dict | None = None,
    train_transform=None,
    val_transform=None,
    test_transform=None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    random_seed: int = 42
):
    """
    Crea i DataLoader per train/val/test a partire da una struttura a cartelle:

        base_path/
            angry/
            happy/
            ...

    Il codice:
    1) legge le classi (sottocartelle)
    2) raccoglie tutti i path immagini e le label
    3) esegue uno split train/val/test con train_test_split e stratify
       per mantenere la stessa distribuzione delle classi in tutti gli split
    4) crea Dataset e DataLoader

    Args:
        base_path: percorso della cartella base contenente le sottocartelle di classe
        batch_size: dimensione batch
        config: config.json caricato (se presente, costruisce automaticamente le transform)
        train_transform/val_transform/test_transform: trasformazioni esplicite (se vuoi forzarle)
        val_ratio: frazione destinata alla validation (es. 0.15)
        test_ratio: frazione destinata al test (es. 0.15)
        shuffle: shuffle del train_loader
        random_seed: seed per split riproducibile (usato se config non lo fornisce)

    Returns:
        train_loader, val_loader, test_loader, classes
    """
    base_path = Path(base_path)

    # 1) Classi: nomi sottocartelle (angry, happy, ...)
    classes = [d.name for d in base_path.iterdir() if d.is_dir()]

    # Mapping classe -> indice numerico
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    # 2) Raccogliamo tutte le immagini e le label
    all_images = []
    all_labels = []
    for cls in classes:
        for img_path in (base_path / cls).glob("*.jpg"):
            all_images.append(img_path)
            all_labels.append(class_to_idx[cls])

    # 3) Seed: se config presente, usiamo config["seed"], altrimenti random_seed
    if config is not None:
        random_seed = config.get("seed", random_seed)

    # Split stratificato (train vs temp)
    # temp = (val + test)
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        all_images,
        all_labels,
        test_size=(val_ratio + test_ratio),
        stratify=all_labels,
        random_state=random_seed
    )

    # Split stratificato (val vs test) all'interno di temp
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs,
        temp_labels,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        stratify=temp_labels,
        random_state=random_seed
    )

    # 4) Trasformazioni:
    # Se hai config, e non fornisci trasformazioni a mano, le costruiamo automaticamente.
    if config is not None:
        if train_transform is None:
            train_transform = build_transforms(config, mode="train")
        if val_transform is None:
            val_transform = build_transforms(config, mode="val")
        if test_transform is None:
            test_transform = build_transforms(config, mode="test")
    else:
        # Fallback: se non c'è config, usiamo train_transform anche per val/test
        if val_transform is None:
            val_transform = train_transform
        if test_transform is None:
            test_transform = train_transform

    # 5) Creazione Dataset
    train_dataset = FER2013Dataset(train_imgs, train_labels, transform=train_transform)
    val_dataset = FER2013Dataset(val_imgs, val_labels, transform=val_transform)
    test_dataset = FER2013Dataset(test_imgs, test_labels, transform=test_transform)

    # 6) Creazione DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, classes
