# Report

## Model Training Experiments

### Experiment 1: Number of Epochs

**Obiettivo:** determinare un numero di epoche sufficiente a garantire la convergenza del modello senza introdurre overfitting e senza spreco computazionale.

Durante gli esperimenti è stato osservato che sia la **CNN Custom** sia il modello **ResNet18 pre-addestrato** raggiungono una stabilizzazione delle prestazioni entro circa **10 epoche**. Oltre tale soglia non si osservano miglioramenti significativi in termini di accuracy di validazione.

**Osservazione:** l’andamento della validation accuracy tende a stabilizzarsi rapidamente, mentre la training accuracy continua a crescere leggermente, indicando l’inizio di overfitting.

**Conclusione:** 10 epoche rappresentano un buon compromesso tra prestazioni e costo computazionale.

---

### Experiment 2: Network Architecture

È stato effettuato un confronto tra:
- **CNN Custom**, progettata manualmente
- **ResNet18**, utilizzata tramite transfer learning e pre-addestrata su ImageNet

Entrambi i modelli sono stati addestrati utilizzando le stesse trasformazioni di preprocessing e le stesse impostazioni di base (batch size, optimizer, loss).

#### Risultati su Test Set

| Modello        | Accuracy |
|---------------|----------|
| CNN Custom    | ~0.96    |
| ResNet18     | ~0.99    |

**Analisi qualitativa:**
- ResNet18 mostra una **convergenza più rapida** e una **migliore capacità di generalizzazione**
- La CNN Custom, pur essendo molto più semplice, raggiunge comunque risultati competitivi
- Il transfer learning risulta particolarmente efficace anche su immagini in scala di grigi

**Conclusione:** ResNet18 supera la CNN Custom su tutte le metriche, ma la CNN rimane un valido baseline didattico.

---

### Experiment 3: Learning Rate

Non è stata condotta una ricerca sistematica del learning rate per limiti di tempo.  
Sono stati utilizzati i seguenti valori:

- **CNN Custom:** LR = 0.001
- **ResNet18:** LR = 0.0001

Queste scelte hanno garantito una convergenza stabile e prestazioni elevate senza instabilità durante l’addestramento.

**Possibili sviluppi:** confronto esplicito tra learning rate diversi (0.001, 0.0005, 0.0001) utilizzando validation accuracy come criterio di selezione.

---

## Dataset & Preprocessing

- **Dataset:** FER2013
- **Numero di classi:** 7 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Formato immagini:** scala di grigi, 48×48 pixel

### Split del dataset
Il dataset è stato suddiviso in:
- 70% training
- 15% validation
- 15% test

Lo split è stato effettuato in modo casuale ma riproducibile tramite seed fisso.

### Trasformazioni
**Training:**
- Resize a 48×48
- Conversione in tensore
- Normalizzazione
- Random Horizontal Flip
- Random Rotation
- Random Resized Crop

**Validation / Test:**
- Resize a 48×48
- Conversione in tensore
- Normalizzazione

---

## Training Setup

- **Optimizer:** Adam
- **Loss:** CrossEntropyLoss con pesi di classe (per gestire dataset sbilanciato)
- **Batch size:** 64
- **Device:** CPU / GPU (CUDA se disponibile)
- **Salvataggio modello:** miglior modello in base alla validation accuracy

---

## Results & Analysis

Le prestazioni finali sono state valutate sul test set utilizzando:
- Accuracy
- Classification Report
- Confusion Matrix

**Osservazioni principali:**
- Entrambi i modelli mostrano buone capacità di generalizzazione
- ResNet18 presenta meno errori nelle classi più difficili
- L’uso di pesi di classe ha contribuito a ridurre l’impatto dello sbilanciamento
- La data augmentation ha migliorato la robustezza del modello

Non sono stati osservati segni evidenti di overfitting grazie all’uso combinato di validation set, dropout e augmentation.

---

## Conclusions

- Sono stati confrontati un **modello custom** e un **modello basato su transfer learning**
- Il transfer learning con ResNet18 ha fornito le migliori prestazioni
- La CNN Custom rappresenta un ottimo punto di partenza e baseline
- Il progetto dimostra una pipeline completa: analisi dati, training, validazione e testing

**Possibili estensioni future:**
- Fine-tuning di più layer di ResNet18
- Utilizzo di architetture più profonde (ResNet50)
- Analisi dettagliata degli errori per classe
- Introduzione di tecniche di early stopping automatico
