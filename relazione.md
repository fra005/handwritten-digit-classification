# Relazione breve – Riconoscimento di cifre con Machine Learning

## 1. Introduzione
Lo scopo del progetto è realizzare un semplice sistema di classificazione capace di riconoscere cifre scritte a mano.  
Il problema viene affrontato come un task di classificazione supervisionata, perché per ogni immagine del dataset è disponibile l'etichetta corretta.

## 2. Dataset
È stato utilizzato il dataset `digits` della libreria scikit-learn.  
Il dataset contiene immagini 8x8 pixel raffiguranti cifre da 0 a 9. Ogni immagine è rappresentata da 64 valori numerici, uno per ogni pixel.

## 3. Preparazione dei dati
I dati sono stati suddivisi in:
- 80% per il training;
- 20% per il test.

È stata usata una suddivisione stratificata, in modo da mantenere una distribuzione simile delle classi nei due insiemi.

## 4. Modelli confrontati
Sono stati utilizzati quattro algoritmi:

1. Logistic Regression
2. K-Nearest Neighbors
3. Random Forest
4. Support Vector Machine

Per Logistic Regression, KNN e SVM è stata applicata anche una standardizzazione delle feature tramite `StandardScaler`.

## 5. Valutazione
La metrica principale utilizzata è l'accuracy, cioè la percentuale di classificazioni corrette sul test set.

Oltre all'accuracy, il programma genera:
- classification report;
- confusion matrix;
- esempi di previsioni effettuate dal modello migliore.

## 6. Risultati
I risultati vengono salvati automaticamente nel file:

`results/model_results.csv`

Il modello con l'accuracy più alta viene selezionato automaticamente e salvato come:

`models/best_model.joblib`

## 7. Conclusioni
Il progetto mostra come diversi algoritmi possano ottenere prestazioni differenti sullo stesso problema.  
Il confronto permette di capire che non esiste un modello migliore in assoluto: la scelta dipende dal dataset, dal tipo di problema e dai criteri utilizzati per valutare le prestazioni.

## 8. Possibili miglioramenti
Come sviluppi futuri sarebbe possibile:
- effettuare il tuning degli iperparametri;
- provare una cross-validation;
- utilizzare MNIST;
- creare una GUI;
- aggiungere una funzione che permetta all'utente di disegnare una cifra e farla classificare dal modello.
