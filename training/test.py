import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report

from data.DataSets.fer_dataset import create_dataloaders
from utils.config_validator import load_and_validate_config
from training.custom_cnn import CustomCNN
from pretrained_model import get_pretrained_model


def build_transform(cfg):
    img_size = cfg["transforms"]["img_size"]
    mean = cfg["transforms"]["normalize"]["mean"]
    std = cfg["transforms"]["normalize"]["std"]

    t = []
    if cfg["transforms"]["val_test"]["resize"]:
        t.append(transforms.Resize((img_size, img_size)))
    t.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(t)


def get_model(cfg, num_classes, device):
    name = cfg["model"]["name"]
    finetune = cfg["model"].get("finetune", False)
    if name.lower().startswith("resnet"):
        model = get_pretrained_model(num_classes=num_classes)
        if finetune:
            for param in model.parameters():
                param.requires_grad = False
            if hasattr(model, "fc"):
                for param in model.fc.parameters():
                    param.requires_grad = True
    else:
        model = CustomCNN(num_classes=num_classes)
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["best", "last"], default="best", help="Which checkpoint to load")
    parser.add_argument("--model", type=str, help="Override model name from config")
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = BASE_DIR / "configs" / "config.json"
    SCHEMA_PATH = BASE_DIR / "configs" / "config_schema.json"

    config, BASE_DIR = load_and_validate_config(CONFIG_PATH, SCHEMA_PATH, base_dir=BASE_DIR, check_paths=True)

    if args.model:
        config["model"]["name"] = args.model

    data_path = BASE_DIR / config["dataset"]["paths"]["test_root"]
    batch_size = config["training"]["batch_size"]
    checkpoint_cfg = config["training"]["checkpoint"]
    models_dir = BASE_DIR / checkpoint_cfg["dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.get("device", "auto") == "auto" else torch.device(config.get("device"))

    transform = build_transform(config)

    _, _, test_loader, classes = create_dataloaders(
        base_path=data_path,
        batch_size=batch_size,
        train_transform=transform,
        augment=False
    )

    model = get_model(config, num_classes=len(classes), device=device)

    # select checkpoint path
    selected = args.which
    if selected == "best":
        model_path = models_dir / checkpoint_cfg["best_name"]
        if not model_path.exists():
            model_path = models_dir / checkpoint_cfg["last_name"]
    else:
        model_path = models_dir / checkpoint_cfg["last_name"]
        if not model_path.exists():
            model_path = models_dir / checkpoint_cfg["best_name"]

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Modello caricato da {model_path}")
    else:
        print(f"⚠ File {model_path} non trovato. Usando modello con pesi iniziali.")

    model.eval()
    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_loss /= total
    test_acc = correct / total

    print("\n=== RISULTATI TEST SET ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}\n")

    print("=== CLASSIFICATION REPORT ===")
    print(classification_report(all_labels, all_preds, target_names=classes))

    print("=== CONFUSION MATRIX ===")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
