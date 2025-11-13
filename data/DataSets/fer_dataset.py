from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

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

        # Definiamo augmentation leggera
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(48, scale=(0.8,1.0))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('L')  # sempre grayscale

        if self.augment and self.augment_transform:
            img = self.augment_transform(img)
        if self.transform:
            img = self.transform(img)

        return img, label


def create_dataloaders(base_path, batch_size=64, train_transform=None, augment=True, val_ratio=0.15, test_ratio=0.15, shuffle=True, random_seed=42):
    """
    Crea train, validation e test DataLoader.
    Args:
        base_path: Path della cartella train di FER2013
        batch_size: batch size
        train_transform: trasformazioni da applicare
        augment: True per augmentation sul train
        val_ratio: percentuale validation
        test_ratio: percentuale test
    Returns:
        train_loader, val_loader, test_loader
    """
    base_path = Path(base_path)
    classes = [d.name for d in base_path.iterdir() if d.is_dir()]
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    # Raccogli immagini e label
    all_images = []
    all_labels = []
    for cls in classes:
        for img_path in (base_path / cls).glob("*.jpg"):
            all_images.append(img_path)
            all_labels.append(class_to_idx[cls])

    # Shuffle e split
    random.seed(random_seed)
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    all_images[:], all_labels[:] = zip(*combined)

    total = len(all_images)
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)
    train_count = total - val_count - test_count

    train_imgs, train_labels = all_images[:train_count], all_labels[:train_count]
    val_imgs, val_labels = all_images[train_count:train_count+val_count], all_labels[train_count:train_count+val_count]
    test_imgs, test_labels = all_images[train_count+val_count:], all_labels[train_count+val_count:]

    # Dataset
    train_dataset = FER2013Dataset(train_imgs, train_labels, transform=train_transform, augment=augment)
    val_dataset = FER2013Dataset(val_imgs, val_labels, transform=train_transform, augment=False)
    test_dataset = FER2013Dataset(test_imgs, test_labels, transform=train_transform, augment=False)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, classes
