"""Model training script.

This file provides a simple classification training pipeline.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Allow running this file directly from any working directory.
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import build_preprocessor, save_preprocessor  # noqa: E402


def train_model(df: pd.DataFrame, target_col: str = "target"):
    """Train a baseline logistic regression classifier.

    Returns the fitted pipeline, test features, and test labels so the
    caller can immediately evaluate without a second data split.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from input data.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    preprocessor = build_preprocessor(X)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def save_model(model, output_path: str | Path = "models/model.joblib") -> None:
    """Save trained model to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


if __name__ == "__main__":  # pragma: no cover
    from evaluate import evaluate_classification, save_metrics

    data_path = Path("data/processed/sample_clean.csv")
    if not data_path.exists():
        print("Processed dataset not found. Run data_preprocessing.py first.")
        sys.exit(1)

    dataset = pd.read_csv(data_path)
    trained_model, X_test, y_test = train_model(dataset, target_col="target")

    save_model(trained_model, "models/baseline_model.joblib")
    print("Model saved to models/baseline_model.joblib")

    save_preprocessor(
        trained_model.named_steps["preprocessor"],
        "models/preprocessor.joblib",
    )
    print("Preprocessor saved to models/preprocessor.joblib")

    metrics = evaluate_classification(trained_model, X_test, y_test)
    save_metrics(metrics, "reports/metrics/baseline_metrics.json")
    print(f"Metrics saved — Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
