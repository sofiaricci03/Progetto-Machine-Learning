"""Unified training script — supporta sia rete custom che pretrained.

Scegli il modello tramite `configs/config.json` (campo `model.name`) o con `--model`.
"""

import sys
from pathlib import Path
import argparse
import time
from datetime import datetime
import json
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from data.DataSets.fer_dataset import create_dataloaders
# Import del modello pre-trained con fallback per esecuzione come modulo o script
try:
    from training.pretrained_model import get_pretrained_model
except Exception:
    try:
        from .pretrained_model import get_pretrained_model
    except Exception:
        from pretrained_model import get_pretrained_model
from training.custom_cnn import CustomCNN
from utils.config_validator import load_and_validate_config


def build_transforms(cfg, mode: str = "train"):
    from torchvision import transforms
    img_size = cfg["transforms"]["img_size"]
    mean = cfg["transforms"]["normalize"]["mean"]
    std = cfg["transforms"]["normalize"]["std"]

    tfms = []
    if mode == "train":
        if cfg["transforms"]["train"]["resize"]:
            tfms.append(transforms.Resize((img_size, img_size)))
        aug = cfg["transforms"]["train"]["augmentation"]
        if aug is not None:
            tfms.append(transforms.RandomResizedCrop(img_size, scale=tuple(aug["random_resized_crop_scale"])))
            if aug.get("horizontal_flip", False):
                tfms.append(transforms.RandomHorizontalFlip())
            if aug.get("rotation_deg", 0) > 0:
                tfms.append(transforms.RandomRotation(aug.get("rotation_deg", 0)))
    else:
        if cfg["transforms"]["val_test"]["resize"]:
            tfms.append(transforms.Resize((img_size, img_size)))

    tfms.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(tfms)


def get_model(cfg, num_classes, device):
    name = cfg["model"]["name"]
    finetune = cfg["model"].get("finetune", False)

    if name.lower().startswith("resnet") or name.lower().startswith("resnet18"):
        model = get_pretrained_model(num_classes=num_classes)
        if finetune:
            # freeze all except final linear layer
            for param in model.parameters():
                param.requires_grad = False
            if hasattr(model, "fc"):
                for param in model.fc.parameters():
                    param.requires_grad = True
    else:
        # default: custom cnn
        model = CustomCNN(num_classes=num_classes)

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Train model (custom or pretrained) using config.json")
    parser.add_argument("--model", type=str, help="Override model name from config")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping even if enabled in config")
    parser.add_argument("--no-tb", action="store_true", dest="no_tb", help="Disable TensorBoard logging")
    parser.add_argument("--epochs", type=int, help="Override number of epochs from config")
    parser.add_argument("--resume", type=str, help="Path to checkpoint file to resume from (overrides config.resume)")
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = BASE_DIR / "configs" / "config.json"
    SCHEMA_PATH = BASE_DIR / "configs" / "config_schema.json"

    config, BASE_DIR = load_and_validate_config(CONFIG_PATH, SCHEMA_PATH, base_dir=BASE_DIR, check_paths=True)

    # allow CLI override
    if args.model:
        config["model"]["name"] = args.model

    data_path = BASE_DIR / config["dataset"]["paths"]["train_root"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"].get("learning_rate", 1e-3)
    wd = config["training"].get("weight_decay", 0.0)
    checkpoint_cfg = config["training"]["checkpoint"]

    models_dir = BASE_DIR / checkpoint_cfg["dir"]
    best_path = models_dir / checkpoint_cfg["best_name"]
    last_path = models_dir / checkpoint_cfg["last_name"]
    (models_dir).mkdir(parents=True, exist_ok=True)

    # CLI overrides
    if args.epochs:
        epochs = args.epochs

    # Device
    dev = config.get("device", "auto")
    if dev == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev)

    # TensorBoard writer (lazy import so tensorboard is optional)
    writer = None
    tb_cfg = config.get("logging", {}).get("tensorboard", {})
    if (not args.no_tb) and tb_cfg.get("enabled", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            print("⚠ TensorBoard non disponibile. Esegui `pip install tensorboard` per abilitare il logging")
            writer = None
        else:
            tb_base = BASE_DIR / tb_cfg.get("log_dir", "runs")
            tb_dir = tb_base / datetime.now().strftime("%Y%m%d-%H%M%S")
            writer = SummaryWriter(str(tb_dir))
            # Log some basic info
            try:
                writer.add_text("config", json.dumps({"model": config.get("model"), "training": config.get("training")}, indent=2))
            except Exception:
                pass

    # transforms & dataloaders
    train_transform = build_transforms(config, mode="train")
    val_transform = build_transforms(config, mode="val_test")

    start_epoch = 1
    # Resume override from CLI
    if args.resume:
        resume_cfg["enabled"] = True
        resume_cfg["path"] = args.resume

    train_loader, val_loader, test_loader, classes = create_dataloaders(
        base_path=data_path,
        batch_size=batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        augment=True
    )

    num_classes = len(classes)
    model = get_model(config, num_classes, device)

    # Criterion with class weights if requested
    base_path = Path(data_path)
    counts = [len(list((base_path / cls).glob("*.jpg"))) for cls in classes]
    if config["training"]["loss"]["class_weights"] == "inverse_frequency":
        total = sum(counts)
        weights = torch.tensor([total / c for c in counts], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    opt_name = config["training"]["optimizer"].lower()
    if opt_name == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)

    # Resume if requested in config (or via --resume)
    resume_cfg = checkpoint_cfg.get("resume", {"enabled": False})
    if resume_cfg.get("enabled", False):
        resume_path = (BASE_DIR / resume_cfg.get("path")).resolve()
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device)
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
                if "optimizer_state" in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt["optimizer_state"])
                    except Exception:
                        print("⚠ Impossibile ripristinare optimizer state (incompatibile)")
                start_epoch = ckpt.get("epoch", 0) + 1
                best_val_acc = ckpt.get("val_acc", 0.0)
                print(f"✓ Resumed model+optimizer from {resume_path} (epoch {start_epoch-1})")
            else:
                model.load_state_dict(ckpt)
                print(f"✓ Resumed model weights from {resume_path}")
        else:
            print(f"⚠ Resume requested but file not found: {resume_path}")

    # Early stopping
    es_cfg = config["training"].get("early_stopping", {})
    es_enabled = es_cfg.get("enabled", False) and not args.no_early_stop
    es_monitor = es_cfg.get("monitor", "val_loss")
    es_mode = es_cfg.get("mode", "min")
    es_patience = es_cfg.get("patience", 5)
    es_min_delta = es_cfg.get("min_delta", 0.0)
    es_target = es_cfg.get("target", {"enabled": False})

    best_metric = float("inf") if es_mode == "min" else float("-inf")
    patience_ctr = 0

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{epochs} | Time {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # TensorBoard logging
        if writer:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/accuracy", train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            # log some parameter histograms every 5 epochs
            if epoch % 5 == 0:
                for name, param in model.named_parameters():
                    if param is not None:
                        writer.add_histogram(f"params/{name}", param.detach().cpu(), epoch)

        # Checkpointing (save model + optimizer state)
        if checkpoint_cfg.get("save_best", False) and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dict = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
            torch.save(save_dict, best_path)
            print(f"✔ Best model saved to {best_path} (Val Acc: {best_val_acc:.4f})")

        if checkpoint_cfg.get("save_last", False):
            save_dict = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
            torch.save(save_dict, last_path)

        # Early stopping logic
        if es_enabled:
            if es_monitor == "val_loss":
                current = val_loss
            else:
                current = val_acc

            improved = (es_mode == "min" and current <= best_metric - es_min_delta) or (es_mode == "max" and current >= best_metric + es_min_delta)
            if improved:
                best_metric = current
                patience_ctr = 0
            else:
                patience_ctr += 1
                print(f"⏳ EarlyStopping: {patience_ctr}/{es_patience} (no improvement)")

            # target stop
            if es_target.get("enabled", False):
                target_metric = es_target.get("value", 1.0)
                if (es_target.get("metric") == "val_accuracy" and val_acc >= target_metric) or (es_target.get("metric") == "val_f1" and False):
                    print(f"🎯 Target reached ({es_target.get('metric')} >= {target_metric}). Stopping.")
                    break

            if patience_ctr >= es_patience:
                print("⛔ Early stopping triggered.")
                break

    # finish
    print("Training finished.")
    if writer:
        writer.flush()
        writer.close()

    print("Training finished.")


if __name__ == "__main__":
    main()
