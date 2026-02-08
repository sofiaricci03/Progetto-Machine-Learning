import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report

from data.DataSets.fer_dataset import create_dataloaders, build_transforms, set_seed
from utils.config_validator import load_and_validate_config
from training.custom_cnn import CustomCNN
# Import del modello pre-trained con fallback per esecuzione come modulo o script
try:
    from training.pretrained_model import get_pretrained_model
except Exception:
    try:
        from .pretrained_model import get_pretrained_model
    except Exception:
        from pretrained_model import get_pretrained_model


def get_model(cfg, num_classes, device):
    #Recupera il nome del modello e se deve fare fine-tuning dal dizionario cfg
    name = cfg["model"]["name"]
    finetune = cfg["model"].get("finetune", False)
    if name.lower().startswith("resnet"):
    # Se il nome inizia con "resnet", usa un modello pre-addestrato
        model = get_pretrained_model(num_classes=num_classes)
       
        if finetune:    # Se fine-tuning è attivo, congela tutti i layer (tecnica che addestra solo l'ultimo layer perché la rete già riconosce forme, bordi, pattern generali) 
            for param in model.parameters():
                param.requires_grad = False
                # Tranne l'ultimo (fc = fully connected)
                # In questo modo, solo l'ultimo strato impara le nuove classi
            if hasattr(model, "fc"):
                for param in model.fc.parameters():
                    param.requires_grad = True
    else:
        #Altrimenti, usa la rete custom semplice definita in custom_cnn.py
        model = CustomCNN(num_classes=num_classes)
    return model.to(device)


# Il main() è la funzione principale che viene eseguita quando si avvia lo script. 
def main():
    # Inizializza il lettore di argomenti da terminale
    parser = argparse.ArgumentParser()
    
    # --which: permette di scegliere se testare il modello "migliore" (best) o l'ultimo salvato (last)
    parser.add_argument("--which", choices=["best", "last"], default="best", help="Which checkpoint to load")
    # --model: permette di cambiare modello al volo senza modificare il file JSON
    parser.add_argument("--model", type=str, help="Override model name from config")
    args = parser.parse_args()

    # Trova la cartella base del progetto risalendo di due livelli rispetto a questo file, e definisce i percorsi per il file di configurazione e lo schema di validazione.
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = BASE_DIR / "configs" / "config.json"
    SCHEMA_PATH = BASE_DIR / "configs" / "config_schema.json"

    # CONFIG
    # Carica e convalida il file JSON. load_and_validate_config si assicura che il file 
    # rispetti lo schema e che i percorsi esistano
    config, BASE_DIR = load_and_validate_config(CONFIG_PATH, SCHEMA_PATH, base_dir=BASE_DIR, check_paths=True)  # apre il file json e lo carica

    # Imposta seed per bloccare la riproducibilità. 
    #In questo modo, ogni volta che esegui lo script, otterrai gli stessi risultati (stesse inizializzazioni dei pesi, stessa suddivisione dei dati, ecc.) se usi lo stesso seed. 
    #Per ottenere risultati diversi, s può cambiare il valore del seed o rimuoverlo
    set_seed(config.get("seed", 42))

    # Se l'utente ha scritto un modello diverso nel terminale, sovrascrive quello presente nel file JSON
    #permette di testare modelli diversi senza dover modificare il file di configurazione ma specificando il nome del modello desiderato con l'argomento --model quando si esegue lo script
    if args.model:
        config["model"]["name"] = args.model

    data_path = BASE_DIR / config["dataset"]["paths"]["test_root"]  # estrae i percorsi
    batch_size = config["training"]["batch_size"]   # Prende la dimensione del batch (quante immagini processare insieme)
    checkpoint_cfg = config["training"]["checkpoint"]   # Recupera la cartella dove sono salvati i file .pth (i pesi del modello)
    models_dir = BASE_DIR / checkpoint_cfg["dir"]

# Sceglie dove eseguire i calcoli:
# Se impostato su 'auto', cerca la GPU (cuda), altrimenti usa la CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.get("device", "auto") == "auto" else torch.device(config.get("device"))  # controlla se gpu è disponibile, se sì la usa

    # LOAD TEST SET
    # Usiamo gli underscore per ignorare i primi due output (train e val loader) perché serve solo il test_loader
    _, _, test_loader, classes = create_dataloaders(
        base_path=data_path,
        batch_size=batch_size,
        config=config  # non vogliamo fare data augmentation sul test set
    )

    # LOAD MODEL
    # Crea un'istanza del modello con il numero di classi corretto
    model = get_model(config, num_classes=len(classes), device=device)

    # select checkpoint path
    # Se l'utente ha scelto "best", cerca il file del modello migliore (best_name) nella cartella dei modelli. Se non lo trova, cerca l'ultimo modello salvato (last_name).
    # Se invece ha scelto "last", cerca prima l'ultimo modello salvato e se non lo trova, cerca il modello migliore. In questo modo, se uno dei due file non esiste, l'altro viene usato come fallback. 
    # Se nessuno dei due esiste, viene stampato un messaggio e si procede con i pesi iniziali del modello (non addestrato).
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
        ckpt = torch.load(model_path, map_location=device)  # carica i pesi salvati del modello dal percorso specificato
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            print(f"Modello caricato da checkpoint {model_path}")
        else:
            model.load_state_dict(ckpt)
            print(f"Modello caricato da {model_path}")
    else:
        print(f"File {model_path} non trovato. Usando modello con pesi iniziali.")

    model.eval()  # disattiva il dropout
    criterion = nn.CrossEntropyLoss()

    # TEST LOOP
    test_loss = 0  # inizializza la variabile per tenere traccia della perdita totale sul test set
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

            all_labels.extend(labels.cpu().numpy())  # sposta le etichette su cpu e le converte in numpy array
            all_preds.extend(predicted.cpu().numpy())  # fa lo stesso con le predizioni

    test_loss /= total
    test_acc = correct / total

    # RESULTS
    # Genera una tabella con tre metriche chiave per ogni emozione
    # precisione: quante volte il modello ha fatto una previsione corretta per una classe specifica
    # recall: quante volte il modello ha identificato correttamente tutti i casi di una classe
    # f1-score: media armonica tra precisione e recall
    print("\n=== RISULTATI TEST SET ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}\n")

    print("=== CLASSIFICATION REPORT ===")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Tabella che mostra i punti in cui il modello ha confuso delle emozioni con altre
    print("=== CONFUSION MATRIX ===")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
