# Riconoscimento di cifre scritte a mano con Machine Learning

## Descrizione
Questo progetto è stato realizzato come esercizio universitario di Machine Learning in Python.

L'obiettivo è classificare immagini di cifre scritte a mano da 0 a 9 utilizzando il dataset `digits` disponibile in scikit-learn.  
Il progetto confronta diversi algoritmi di classificazione e valuta quale ottiene le prestazioni migliori.

## Obiettivi
- caricare ed esplorare un dataset reale;
- preparare i dati per l'addestramento;
- dividere i dati in training set e test set;
- addestrare più modelli di Machine Learning;
- confrontare le prestazioni dei modelli;
- visualizzare una confusion matrix;
- salvare il modello migliore.

## Algoritmi utilizzati
Il progetto confronta:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Support Vector Machine (SVM)

## Struttura del progetto

```text
progetto_ml_cifre/
│
├── main.py
├── requirements.txt
├── README.md
├── relazione.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualization.py
│
├── notebooks/
│   └── analisi.ipynb
│
├── results/
└── models/
```

## Dataset
Viene utilizzato il dataset `digits` di scikit-learn.

Ogni esempio è un'immagine 8x8 pixel di una cifra scritta a mano.  
Ogni immagine viene trasformata in un vettore di 64 valori numerici.

## Installazione

È consigliato creare un ambiente virtuale:

```bash
python -m venv .venv
```

Su Windows:

```bash
.venv\Scripts\activate
```

Su macOS/Linux:

```bash
source .venv/bin/activate
```

Installare poi le librerie necessarie:

```bash
pip install -r requirements.txt
```

## Avvio del progetto

Eseguire:

```bash
python main.py
```

Il programma:
1. carica il dataset;
2. divide i dati in training e test set;
3. addestra quattro modelli;
4. confronta le accuracy;
5. salva i risultati in `results/model_results.csv`;
6. genera la confusion matrix del modello migliore;
7. salva il modello migliore nella cartella `models/`.

## Competenze utilizzate
- Python
- scikit-learn
- pandas
- matplotlib
- Machine Learning supervisionato
- classificazione
- train/test split
- model evaluation
- Git / GitLab

## Possibili sviluppi futuri
Il progetto potrebbe essere esteso:
- testando altri algoritmi;
- facendo hyperparameter tuning;
- creando una piccola interfaccia grafica;
- permettendo all'utente di disegnare una cifra e ottenere la previsione del modello;
- utilizzando un dataset più grande come MNIST.
