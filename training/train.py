"""Unified training script — supporta sia rete custom che pretrained.

Scegli il modello tramite `configs/config.json` (campo `model.name`) o con `--model`.
"""

import sys  # Per manipolare il percorso di ricerca dei moduli Python
from pathlib import Path  # Per gestire i percorsi dei file in modo cross-platform (Windows/Linux/Mac)
import argparse  # Per leggere argomenti passati da riga di comando (es. --model, --epochs)
import time  # Per misurare il tempo di esecuzione di ogni epoca
from datetime import datetime  # Per creare timestamp univoci per i log di TensorBoard
import json  # Per convertire configurazioni in formato JSON (per TensorBoard)

# Aggiunge la cartella radice del progetto al percorso di ricerca dei moduli
# Permette di importare moduli con "from data.DataSets..." anche eseguendo lo script da altre cartelle
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  
import torch.nn as nn  
import torch.optim as optim  
from data.DataSets.fer_dataset import create_dataloaders, build_transforms, set_seed  


# Import del modello pre-trained con fallback per esecuzione come modulo o script
# Questo blocco try-except permette di importare il modello pre-trained in modi diversi
# a seconda di come viene eseguito lo script (come modulo, come script, da IDE, ecc.)
try:
    # Prova prima ad importare come modulo del package training
    from training.pretrained_model import get_pretrained_model
except Exception:
    try:
        # Se fallisce, prova import relativo (quando eseguito come modulo)
        from .pretrained_model import get_pretrained_model
    except Exception:
        # Ultima possibilità: import diretto (quando eseguito come script)
        from pretrained_model import get_pretrained_model

# Importa la rete custom semplice definita nel progetto
from training.custom_cnn import CustomCNN
# Importa la funzione per caricare e validare il file di configurazione JSON
from utils.config_validator import load_and_validate_config


def get_model(cfg, num_classes, device):
    """
    Funzione che crea e restituisce il modello di rete neurale da utilizzare.
    
    Args:
        cfg: dizionario di configurazione caricato da config.json
        num_classes: numero di classi da classificare (es. 7 emozioni)
        device: dispositivo su cui eseguire il modello (CPU o GPU)
    
    Returns:
        model: istanza del modello pronta per il training, già spostata sul device corretto
    """
    # Legge il nome del modello dal dizionario cfg (es. "cnn_simple" o "resnet18")
    name = cfg["model"]["name"]
    # Verifica se è richiesto il fine-tuning (default: False se non specificato)
    finetune = cfg["model"].get("finetune", False)

    # Controlla se il nome del modello inizia con "resnet" (modello pre-addestrato)
    if name.lower().startswith("resnet") or name.lower().startswith("resnet18"):
        # Crea un'istanza di ResNet pre-addestrato (già addestrato su ImageNet)
        model = get_pretrained_model(num_classes=num_classes)
        
        if finetune:
            # Se fine-tuning è attivo, congela tutti i layer della rete
            # freeze all except final linear layer
            for param in model.parameters():
                # requires_grad=False significa "non aggiornare questi pesi durante il training"
                param.requires_grad = False
            
            # Scongela solo l'ultimo layer fully connected (fc)
            # hasattr verifica se il modello ha l'attributo "fc" (layer finale)
            if hasattr(model, "fc"):
                for param in model.fc.parameters():
                    # requires_grad=True: solo questo layer imparerà le nuove classi
                    param.requires_grad = True
    else:
        # default: custom cnn
        # Se non è un modello ResNet, usa la rete custom definita nel progetto
        model = CustomCNN(num_classes=num_classes)

    # Sposta il modello sul dispositivo specificato (GPU se disponibile, altrimenti CPU)
    # .to(device) copia tutti i parametri del modello sulla memoria del device
    return model.to(device)


def main():
    """
    Funzione principale che orchestra l'intero processo di addestramento.
    Viene eseguita quando si lancia lo script con: python training/train.py
    """
    # === PARSING ARGOMENTI DA RIGA DI COMANDO ===
    # Crea un parser per leggere gli argomenti passati da terminale
    parser = argparse.ArgumentParser(description="Train model (custom or pretrained) using config.json")
    
    # Definisce gli argomenti che l'utente può passare:
    # --model: per sovrascrivere il modello specificato nel JSON
    parser.add_argument("--model", type=str, help="Override model name from config")
    # --no-early-stop: per disabilitare l'early stopping anche se abilitato nel config
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping even if enabled in config")
    # --no-tb: per disabilitare il logging su TensorBoard
    parser.add_argument("--no-tb", action="store_true", dest="no_tb", help="Disable TensorBoard logging")
    # --epochs: per sovrascrivere il numero di epoche dal config
    parser.add_argument("--epochs", type=int, help="Override number of epochs from config")
    # --resume: per riprendere il training da un checkpoint salvato
    parser.add_argument("--resume", type=str, help="Path to checkpoint file to resume from (overrides config.resume)")
    
    # Analizza gli argomenti passati e li salva nell'oggetto args
    args = parser.parse_args()

    # === CARICAMENTO CONFIGURAZIONE ===
    # Trova la cartella base del progetto (due livelli sopra questo file)
    # __file__ = percorso di questo file train.py
    # .resolve() = converte in percorso assoluto
    # .parent.parent = sale di due livelli (training/ -> progetto/)
    BASE_DIR = Path(__file__).resolve().parent.parent
    # Definisce i percorsi ai file di configurazione
    CONFIG_PATH = BASE_DIR / "configs" / "config.json"  # File con le impostazioni
    SCHEMA_PATH = BASE_DIR / "configs" / "config_schema.json"  # Schema di validazione

    # Carica e valida il file di configurazione JSON
    # Questa funzione:
    # 1. Legge il file config.json
    # 2. Verifica che rispetti lo schema definito
    # 3. Controlla che i percorsi specificati esistano (se check_paths=True)
    # 4. Restituisce il dizionario config e il BASE_DIR aggiornato
    config, BASE_DIR = load_and_validate_config(CONFIG_PATH, SCHEMA_PATH, base_dir=BASE_DIR, check_paths=True)

    # Imposta seed per riproducibilità
    # Imposta il seed per tutti i generatori di numeri casuali (random, numpy, torch)
    # In questo modo, ogni esecuzione darà gli stessi risultati se si usa lo stesso seed
    # Utile per debugging e per poter riprodurre esperimenti
    set_seed(config.get("seed", 42))  # Usa 42 come default se seed non è specificato

    # allow CLI override
    # Se l'utente ha specificato --model dal terminale, sovrascrive il valore del JSON
    if args.model:
        config["model"]["name"] = args.model

    # === ESTRAZIONE PARAMETRI DI TRAINING DAL CONFIG ===
    # Costruisce il percorso completo alla cartella dei dati di training
    data_path = BASE_DIR / config["dataset"]["paths"]["train_root"] 
    # Estrae la dimensione del batch (quante immagini elaborare contemporaneamente)
    batch_size = config["training"]["batch_size"]  # es. 64
    # Numero di epoche (quante volte il modello vedrà l'intero dataset)
    epochs = config["training"]["epochs"]  # es. 30
    # Learning rate: quanto "velocemente" il modello impara (passo di aggiornamento dei pesi)
    lr = config["training"].get("learning_rate", 1e-3)  # Default 0.001 se non specificato
    # Weight decay: regolarizzazione L2 per evitare overfitting
    wd = config["training"].get("weight_decay", 0.0)  # Default 0 (nessuna regolarizzazione)
    # Configurazione per il salvataggio dei checkpoint (modelli salvati)
    checkpoint_cfg = config["training"]["checkpoint"]

    # === PERCORSI PER SALVARE I MODELLI ===
    # Costruisce i percorsi dove salvare i modelli addestrati (dipendendo dal modello scelto)
    models_dir = BASE_DIR / checkpoint_cfg["dir"] 

    model_tag = config["model"]["name"].lower()
    best_path = models_dir / f"best_{model_tag}.pth"
    last_path = models_dir / f"last_{model_tag}.pth"

    # Crea la cartella models se non esiste
    # parents=True: crea anche le cartelle intermedie se necessario
    # exist_ok=True: non da errore se la cartella esiste già
    (models_dir).mkdir(parents=True, exist_ok=True)

    # CLI overrides
    # Se l'utente ha specificato --epochs dal terminale, sovrascrive il valore del JSON
    if args.epochs:
        epochs = args.epochs

    # Device
    # Determina se utilizzare GPU (CUDA) o CPU per il training
    dev = config.get("device", "auto")  # Legge impostazione dal config (default: "auto")
    if dev == "auto":
        # Se "auto", controlla automaticamente se GPU è disponibile
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # usa la GPU se disponibile, altrimenti la CPU
    else:
        # Altrimenti usa il device specificato manualmente (es. "cpu" o "cuda")
        device = torch.device(dev)

    # === TENSORBOARD WRITER ===
    # TensorBoard è uno strumento per visualizzare grafici di loss, accuracy, ecc. durante il training
    # TensorBoard writer (lazy import so tensorboard is optional)
    writer = None  # Inizialmente None (logging disabilitato)
    tb_dir = None  # Percorso dove salvare i log di TensorBoard
    # Legge la configurazione di TensorBoard dal config
    tb_cfg = config.get("logging", {}).get("tensorboard", {})
    
    # Abilita TensorBoard solo se:
    # 1. L'utente NON ha passato --no-tb
    # 2. TensorBoard è abilitato nel config
    if (not args.no_tb) and tb_cfg.get("enabled", False):
        try:
            # importae TensorBoard 
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            # Se l'import fallisce, avvisa l'utente
            print("TensorBoard non disponibile. Esegui `pip install tensorboard` per abilitare il logging")
            writer = None  # Disabilita il logging
        else:
            # Se l'import ha successo, crea la cartella per i log
            tb_base = BASE_DIR / tb_cfg.get("log_dir", "runs")  # es. "Progetto-ML/runs"
            # Crea una sottocartella con timestamp univoco per questa esecuzione
            # es. "runs/20260208-143025" (anno-mese-giorno-ora-minuto-secondo)
            tb_dir = tb_base / datetime.now().strftime("%Y%m%d-%H%M%S")
            # Crea il writer che scriverà i log in quella cartella
            writer = SummaryWriter(str(tb_dir))
            
            # Log some basic info
            # Salva la configurazione del modello e training in TensorBoard per riferimento futuro
            try:
                # Converte parte del config in JSON e lo salva come testo in TensorBoard
                writer.add_text("config", json.dumps({"model": config.get("model"), "training": config.get("training")}, indent=2))
            except Exception:
                # Se non riesce a salvare il config, continua comunque (non è critico)
                pass

    # === CREAZIONE DATALOADER ===
    # Creazione DataLoader
    # Passa il percorso dei dati, la dimensione del batch e le trasformazioni
    start_epoch = 1  # Epoca da cui partire (1 se nuovo training, può essere > 1 se resume)
    
    # create_dataloaders crea tre DataLoader: train, validation e test
    # Un DataLoader è un iteratore che restituisce batch di dati pronti per l'addestramento
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        base_path=data_path,  # Percorso ai dati
        batch_size=batch_size,  # Quante immagini per batch
        config=config  # Config con trasformazioni, augmentation, split ratio, ecc.
    )

    # Conta quante classi ci sono (es. 7 emozioni)
    num_classes = len(classes)
    # Crea il modello (CustomCNN o ResNet) e lo sposta sul device (GPU/CPU)
    model = get_model(config, num_classes, device)

    # === FUNZIONE DI LOSS CON PESI PER CLASSI SBILANCIATE ===
    # Loss con pesi per classi sbilanciate
    # Trasforma una stringa in un oggetto Path per gestire i percorsi dei file
    base_path = Path(data_path)
    # Conta quante immagini ci sono in ogni cartella (una per classe)
    # es. counts = [3000, 500, 2000, ...] se ci sono 3000 immagini "angry", 500 "disgust", ecc.
    counts = [len(list((base_path / cls).glob("*.jpg"))) for cls in classes]  # conta quante immagini ci sono nella cartella di ogni classe
    
    # Se nel config è specificato di usare pesi per le classi
    if config["training"]["loss"]["class_weights"] == "inverse_frequency":
        total = sum(counts)  # somma il totale delle immagini nel db (es. 28709)
        # Calcola peso per ogni classe: peso = totale / numero_immagini_classe
        # Classi con poche immagini avranno peso alto, classi con molte immagini avranno peso basso
        # es. se 'disgust' ha 500 immagini: peso = 28709/500 = 57.4 (peso alto!)
        # es. se 'happy' ha 8000 immagini: peso = 28709/8000 = 3.6 (peso basso)
        weights = torch.tensor([total / c for c in counts], dtype=torch.float32).to(device)  # questo calcolo assegna un peso maggiore alle classi con meno immagini: poche immagini = peso alto; converte la lista di pesi in un tensore PyTorch e lo sposta sulla GPU se disponibile
        # Crea la funzione di loss con pesi
        # Durante il training, errori sulle classi rare conteranno di più
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        # Altrimenti usa CrossEntropyLoss standard (tutte le classi hanno peso uguale)
        criterion = nn.CrossEntropyLoss()

    # === OTTIMIZZATORE ===
    # Ottimizzatore
    # Utilizza Adam, un algoritmo di ottimizzazione che adatta i tassi di apprendimento per ogni parametro
    opt_name = config["training"]["optimizer"].lower()  # Legge il nome (es. "adam", "sgd")
    
    if opt_name == "adam":
        # Adam (Adaptive Moment Estimation): ottimizzatore molto usato, adatta lr per ogni parametro
        # filter(...) seleziona solo i parametri che devono essere aggiornati (requires_grad=True)
        # lr = learning rate (quanto velocemente impara)
        # weight_decay = regolarizzazione L2 (penalizza pesi troppo grandi per evitare overfitting)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        # SGD (Stochastic Gradient Descent): ottimizzatore classico
        # momentum=0.9: usa una "memoria" dei gradienti precedenti per accelerare convergenza
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        # Default: usa Adam se l'ottimizzatore specificato non è riconosciuto
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)

    # === RESUME TRAINING DA CHECKPOINT (OPZIONALE) ===
    # Resume if requested in config (or via --resume)
    # Permette di riprendere il training da dove era stato interrotto
    resume_cfg = checkpoint_cfg.get("resume", {"enabled": False})  # Legge config resume
    
    # Resume override from CLI
    # Se l'utente vuole riprendere da un checkpoint via --resume, aggiorna la config
    if args.resume:
        resume_cfg["enabled"] = True  # Abilita il resume
        resume_cfg["path"] = args.resume  # Usa il percorso specificato dall'utente

    if resume_cfg.get("enabled", False):  # Se resume è abilitato
        # Costruisce il percorso completo al file checkpoint
        resume_path = (BASE_DIR / resume_cfg.get("path")).resolve()
        
        if resume_path.exists():  # Se il file esiste
            # Carica il checkpoint (contiene pesi del modello, stato optimizer, epoca, ecc.)
            ckpt = torch.load(resume_path, map_location=device)
            
            # Verifica se è un checkpoint completo (dizionario) o solo pesi
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                # Carica i pesi del modello
                model.load_state_dict(ckpt["model_state"])
                
                # Prova a caricare anche lo stato dell'optimizer
                if "optimizer_state" in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt["optimizer_state"])
                    except Exception:
                        # Se lo stato è incompatibile (es. diverso optimizer), continua senza
                        print("Impossibile ripristinare optimizer state (incompatibile)")
                
                # Riprende dall'epoca successiva a quella salvata
                start_epoch = ckpt.get("epoch", 0) + 1  # es. se salvato a epoca 10, riprende da 11
                # Recupera la migliore accuracy raggiunta fino a quel momento
                best_val_acc = ckpt.get("val_acc", 0.0)
                print(f"Resumed model+optimizer from {resume_path} (epoch {start_epoch-1})")
            else:
                # Se è solo un file con pesi (senza metadata), carica solo quelli
                model.load_state_dict(ckpt)
                print(f"Resumed model weights from {resume_path}")
        else:
            # Se il file non esiste, avvisa ma continua con training da zero
            print(f"Resume requested but file not found: {resume_path}")

    # === EARLY STOPPING CONFIGURATION ===
    # Early stopping: ferma il training se il modello smette di migliorare
    # Evita di sprecare tempo se il modello ha già raggiunto il suo massimo potenziale
    es_cfg = config["training"].get("early_stopping", {})  # Legge config early stopping
    # Abilita solo se: 1) config dice "enabled": true E 2) utente NON ha passato --no-early-stop
    es_enabled = es_cfg.get("enabled", False) and not args.no_early_stop
    # Quale metrica monitorare (es. "val_loss" o "val_accuracy")
    es_monitor = es_cfg.get("monitor", "val_loss")
    # Mode: "min" se vogliamo minimizzare (loss), "max" se vogliamo massimizzare (accuracy)
    es_mode = es_cfg.get("mode", "min")
    # Patience: quante epoche aspettare senza miglioramenti prima di fermarsi
    es_patience = es_cfg.get("patience", 5)  # es. se 5, aspetta 5 epoche
    # Min delta: miglioramento minimo per considerare che ci sia stato progresso
    es_min_delta = es_cfg.get("min_delta", 0.0)  # es. 0.001 = deve migliorare di almeno 0.1%
    # Target: opzionalmente, ferma se si raggiunge un obiettivo specifico
    es_target = es_cfg.get("target", {"enabled": False})

    # Se early stopping è abilitato, stampa info
    if es_enabled:
        print(f"\n✓ Early stopping abilitato: monitor={es_monitor}, mode={es_mode}, patience={es_patience}, min_delta={es_min_delta}")
        if es_target.get("enabled", False):
            # Se target stopping è abilitato, stampa anche quello
            print(f"  Target stopping: {es_target.get('metric')} >= {es_target.get('value')}")
    
    # Inizializza la migliore metrica vista fino ad ora
    # Se mode="min" (loss): inizia con infinito (qualsiasi valore sarà migliore)
    # Se mode="max" (accuracy): inizia con -infinito (qualsiasi valore sarà migliore)
    best_metric = float("inf") if es_mode == "min" else float("-inf")
    # Contatore di patience: quante epoche consecutive senza miglioramento
    patience_ctr = 0

    # === TRAINING LOOP PRINCIPALE ===
    # Training loop
    # Variabile per tenere traccia della migliore accuratezza di validazione
    # Serve per salvare solo il modello migliore (non l'ultimo, che potrebbe essere peggiore)
    best_val_acc = 0.0

    # Ciclo principale: itera per ogni epoca
    for epoch in range(1, epochs + 1):  # ciclo di addestramento per il numero di epochs specificati (es. da 1 a 30)
        # === FASE DI TRAINING ===
        model.train()  # modalità addestramento - abilita dropout, batch norm, ecc.
        # Inizializza le metriche per questa epoca
        running_loss = 0.0  # Accumula la loss totale
        correct = 0  # Conta predizioni corrette
        total = 0  # Conta numero totale di immagini viste

        start_time = time.time()  # Timestamp inizio epoca (per calcolare durata)
        
        # Itera su tutti i batch nel training set
        for images, labels in train_loader:
            # Ogni iterazione:
            # - images: batch di immagini (es. tensor di shape [64, 1, 48, 48])
            # - labels: etichette corrispondenti (es. tensor [64] con valori 0-6 per le 7 emozioni)
            images, labels = images.to(device), labels.to(device)  # sposta i dati sulla GPU; se non si spostasse, il computer cercherebbe di calcolare i dati in due posti diversi, causando un errore
            optimizer.zero_grad()  # azzera i gradienti calcolati nel passo precedente
            outputs = model(images)  # passa le immagini nel modello e la rete restituisce dei punteggi per ogni classe
            loss = criterion(outputs, labels)  # calcola la loss confrontando i punteggi previsti con le etichette reali per vedere quanto scarto di errore c'è
            loss.backward()  # il modello torna indietro nei livelli per capire chi ha causato l'errore e calcola i gradienti
            optimizer.step()  # aggiorna i pesi del modello in base ai gradienti calcolati per ridurre l'errore successivamente

            # Calcolo delle statistiche in realtime
            running_loss += loss.item() * images.size(0)  # somma la loss (l'errore) moltiplicata per il numero di immagini nel batch corrente
            _, preds = outputs.max(1)  # ottiene le classi previste selezionando l'indice con il punteggio più alto per ogni immagine
            total += labels.size(0)  # aggiorna il contatore totale delle immagini
            correct += (preds == labels).sum().item()  # confronta le classi previste con le etichette reali e conta quante sono corrette

        train_loss = running_loss / total  # calcola media della loss per l'epoch
        train_acc = correct / total  # calcola accuratezza di addestramento (più è vicina a 1.0, meglio è)

        # Validation
        # Il modello viene testato su dati che non ha mai visto prima per valutare le sue prestazioni
        model.eval()  # modalità valutazione, quindi disabilita dropout e batch norm (che servivano in fase addestramento)
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # disabilita il calcolo dei gradienti per risparmiare memoria e velocizzare i calcoli
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)  # passa le immagini nel modello per ottenere le previsioni
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)  # ottiene le classi previste selezionando l'indice con il punteggio più alto per ogni immagine
                val_total += labels.size(0)  # aggiorna il contatore totale delle immagini di validazione
                val_correct += (preds == labels).sum().item()  # confronta le classi previste con le etichette reali e conta quante sono corrette
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total  # calcola la percentuale di precisione finale

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

        # Salvataggio modello migliore
        # Valuta se l'accuratezza ottenuta è la migliore fino ad ora
        if checkpoint_cfg.get("save_best", False) and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dict = {
                "epoch": epoch,
                "model_state": model.state_dict(),  # salva i pesi del modello
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
            torch.save(save_dict, best_path)
            print(f"✔ Best model saved to {best_path} (Val Acc: {best_val_acc:.4f})")
            # Anche se alla fine dell'addestramento, ottengo un modello non performante, avrò sempre salvato il migliore durante il processo

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
            # Determina il valore corrente della metrica da monitorare
            if es_monitor == "val_loss":
                current = val_loss
            elif es_monitor == "val_accuracy":
                current = val_acc
            else:
                current = val_acc  # fallback

            # Verifica se c'è stato miglioramento
            improved = False
            if es_mode == "min":
                improved = current <= best_metric - es_min_delta
            else:  # max
                improved = current >= best_metric + es_min_delta
            
            if improved:
                best_metric = current
                patience_ctr = 0
            else:
                patience_ctr += 1
                print(f"  EarlyStopping: {patience_ctr}/{es_patience} (no improvement in {es_monitor})")

            # Verifica target stopping
            if es_target.get("enabled", False):
                target_metric_name = es_target.get("metric", "val_accuracy")
                target_value = es_target.get("value", 1.0)
                
                # Mappa il nome della metrica al suo valore
                metric_map = {
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "train_loss": train_loss
                }
                
                if target_metric_name in metric_map:
                    current_target_value = metric_map[target_metric_name]
                    # Determina se il target è stato raggiunto in base al tipo di metrica
                    target_reached = False
                    if "loss" in target_metric_name:
                        target_reached = current_target_value <= target_value
                    else:  # accuracy o altre metriche dove più alto è meglio
                        target_reached = current_target_value >= target_value
                    
                    if target_reached:
                        print(f"\nTarget raggiunto: {target_metric_name}={current_target_value:.4f} (target: {target_value:.4f})")
                        print("  Arresto anticipato dell'addestramento.")
                        break

            # Verifica patience esaurita
            if patience_ctr >= es_patience:
                print(f"\nEarly stopping attivato dopo {patience_ctr} epoche senza miglioramento.")
                print(f"  Best {es_monitor}: {best_metric:.4f}")
                break

    # Fine addestramento
    print("\nAddestramento completato.")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    
    if writer:
        writer.flush()
        writer.close()
        if tb_dir:
            print(f"  TensorBoard logs salvati in: {tb_dir}")


if __name__ == "__main__":
    main()
