from pathlib import Path
import joblib

from src.data import load_data
from src.train import get_models, train_models
from src.evaluate import evaluate_models, print_best_model_report
from src.visualization import save_confusion_matrix, save_example_predictions


def main():
    print("=" * 60)
    print("RICONOSCIMENTO DI CIFRE SCRITTE A MANO")
    print("Progetto di Machine Learning in Python")
    print("=" * 60)

    # 1. Caricamento dei dati
    digits, X_train, X_test, y_train, y_test = load_data()

    print(f"\nNumero totale di esempi: {len(digits.data)}")
    print(f"Training set: {len(X_train)} esempi")
    print(f"Test set: {len(X_test)} esempi")
    print(f"Numero di feature per immagine: {X_train.shape[1]}")

    # 2. Creazione e training dei modelli
    models = get_models()
    trained_models = train_models(models, X_train, y_train)

    # 3. Valutazione
    results_df = evaluate_models(trained_models, X_test, y_test)

    print("\nRisultati")
    print("-" * 60)
    print(results_df.to_string(index=False))

    # 4. Salvataggio dei risultati
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results_df.to_csv(
        results_dir / "model_results.csv",
        index=False
    )

    # 5. Selezione del modello migliore
    best_model_name = results_df.iloc[0]["Modello"]
    best_model = trained_models[best_model_name]

    print(f"\nModello migliore: {best_model_name}")
    print(f"Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")

    # 6. Report dettagliato
    print_best_model_report(
        best_model_name,
        best_model,
        X_test,
        y_test
    )

    # 7. Confusion matrix
    save_confusion_matrix(
        best_model,
        X_test,
        y_test,
        results_dir / "confusion_matrix.png"
    )

    # 8. Alcune previsioni di esempio
    save_example_predictions(
        digits,
        best_model,
        results_dir / "example_predictions.png"
    )

    # 9. Salvataggio del modello
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(
        best_model,
        models_dir / "best_model.joblib"
    )

    print("\nFile creati:")
    print("- results/model_results.csv")
    print("- results/confusion_matrix.png")
    print("- results/example_predictions.png")
    print("- models/best_model.joblib")


if __name__ == "__main__":
    main()
