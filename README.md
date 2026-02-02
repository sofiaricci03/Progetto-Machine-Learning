# Progetto-Machine-Learning FER2013 – Facial Expression Recognition with PyTorch

Questo progetto ha l’obiettivo di sviluppare e confrontare diverse architetture di reti neurali per il problema di **Facial Expression Recognition (FER)** utilizzando il dataset **FER2013**.  
L’intero progetto è stato realizzato in **PyTorch** e comprende analisi del dataset, addestramento, validazione e testing dei modelli.


## 📊 Dataset

Il dataset utilizzato è **FER2013**, composto da immagini facciali in **scala di grigi** con risoluzione **48×48** pixel.  
Le immagini sono suddivise in 7 classi emozionali:

- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

Il dataset presenta un **forte sbilanciamento tra le classi**, affrontato tramite:
- analisi statistica della distribuzione
- utilizzo di **pesi di classe** nella funzione di loss
- tecniche di **data augmentation**


## 🔍 Analisi del Dataset

L’analisi esplorativa del dataset è contenuta nel notebook `DataSet_Analysis.ipynb` e comprende:

- visualizzazione di immagini campione per ciascuna classe
- conteggio e distribuzione delle immagini
- calcolo di percentuali, media e deviazione standard
- split stratificato in:
  - 70% training
  - 15% validation
  - 15% test
- definizione delle trasformazioni e delle tecniche di augmentation


## 🔄 Preprocessing e Data Augmentation

Le immagini vengono preprocessate con le seguenti trasformazioni:

- conversione in scala di grigi
- resize a 48×48
- conversione in tensore
- normalizzazione

Durante il training viene applicata **data augmentation** leggera:

- Random Horizontal Flip
- Random Rotation
- Random Resized Crop


## 🧠 Modelli Implementati

### 🔹 CNN Custom

Rete neurale convoluzionale sviluppata da zero:
- 2 blocchi Conv2D + ReLU + MaxPooling
- Fully Connected con Dropout
- Addestramento completo da zero

Questo modello rappresenta il **baseline** del progetto.


### 🔹 Transfer Learning – ResNet18

Modello basato su **ResNet18 pre-addestrata su ImageNet**:
- adattamento dell’input da RGB a grayscale
- congelamento dei layer iniziali
- fine-tuning dell’ultimo blocco convoluzionale
- classificatore finale adattato alle 7 classi FER2013


## 🏋️ Addestramento

Durante l’addestramento:
- viene utilizzato un validation set
- vengono testati diversi iperparametri
- viene salvato automaticamente il **modello con miglior validation accuracy**

### Addestramento CNN Custom
```bash
python train_custom.py

### Addestramento Transfer Learning

python train_pretrained.py

#🧪 Testing e Valutazione

I modelli vengono valutati sul test set tramite:

Accuracy

Classification Report

Confusion Matrix

Test CNN Custom
python test_custom.py

Test Transfer Learning
python test_pretrained.py

⚙️ Tecnologie Utilizzate

Python

PyTorch

Torchvision

NumPy

Pandas

Matplotlib

Scikit-learn

Pillow

📦 Installazione

Installare le dipendenze tramite:

pip install -r requirements.txt

