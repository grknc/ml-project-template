"""Evaluation utilities for classification models."""

import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def evaluate_classification(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute classification metrics (binary and multiclass).

    Returns a dict with accuracy, roc_auc, and a per-class report.
    roc_auc is None when the model does not support probability estimates.
    """
    predictions = model.predict(X_test)
    n_classes = y_test.nunique()

    roc_auc = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        if n_classes == 2:
            roc_auc = roc_auc_score(y_test, probabilities[:, 1])
        else:
            roc_auc = roc_auc_score(
                y_test, probabilities, multi_class="ovr", average="macro"
            )
    else:
        warnings.warn(
            "Model does not support predict_proba — roc_auc will be None.",
            UserWarning,
            stacklevel=2,
        )

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc,
        "classification_report": classification_report(
            y_test, predictions, output_dict=True
        ),
    }
    return metrics


def save_metrics(metrics: dict, output_path: str | Path = "reports/metrics/metrics.json") -> None:
    """Persist metrics to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).parent))

    model_path = Path("models/baseline_model.joblib")
    data_path = Path("data/processed/sample_clean.csv")

    if not model_path.exists():
        print("Model not found. Run train.py first.")
        sys.exit(1)
    if not data_path.exists():
        print("Processed dataset not found. Run data_preprocessing.py first.")
        sys.exit(1)

    from sklearn.model_selection import train_test_split

    dataset = pd.read_csv(data_path)
    target_col = "target"
    X = dataset.drop(columns=[target_col])
    y = dataset[target_col]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = joblib.load(model_path)
    metrics = evaluate_classification(model, X_test, y_test)
    save_metrics(metrics, "reports/metrics/baseline_metrics.json")

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, model.predict(X_test)))
    print("Metrics saved to reports/metrics/baseline_metrics.json")