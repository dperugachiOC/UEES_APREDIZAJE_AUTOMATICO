import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_models(models, X_test, y_test):
    """
    Evalúa múltiples modelos y devuelve métricas y matrices de confusión.
    """

    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        results.append({
            "Modelo": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
            "ConfusionMatrix": confusion_matrix(y_test, y_pred)
        })

    return results


def results_to_dataframe(results):
    """
    Convierte la lista de resultados en un DataFrame ordenado por F1-score.
    """
    df = pd.DataFrame(results)
    df_sorted = df.sort_values("F1-score", ascending=False).reset_index(drop=True)
    return df_sorted


def print_confusion_matrices(results):
    """
    Imprime matrices de confusión en texto (útil para consola o logs).
    """
    for r in results:
        print(f"\nMatriz de confusión - {r['Modelo']}")
        print(r["ConfusionMatrix"])
