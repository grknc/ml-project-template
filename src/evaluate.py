"""Evaluation utilities for classification models."""

from pathlib import Path

import json
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def evaluate_classification(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute basic binary classification metrics."""
    predictions = model.predict(X_test)

    # ROC-AUC requires probability estimates.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probabilities)
    else:
        roc_auc = None

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc,
    }
    return metrics


def save_metrics(metrics: dict, output_path: str | Path = "reports/metrics/metrics.json") -> None:
    """Persist metrics to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    # Placeholder for future CLI usage.
    print("Use evaluate_classification(...) from notebooks or scripts.")
