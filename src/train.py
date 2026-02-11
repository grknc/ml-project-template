"""Model training script.

This file provides a simple classification training pipeline.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from feature_engineering import build_preprocessor


def train_model(df: pd.DataFrame, target_col: str = "target"):
    """Train a baseline logistic regression classifier."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from input data.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    preprocessor = build_preprocessor(df, target_col=target_col)

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


if __name__ == "__main__":
    data_path = Path("data/processed/sample_clean.csv")
    if data_path.exists():
        dataset = pd.read_csv(data_path)
        trained_model, _, _ = train_model(dataset, target_col="target")
        save_model(trained_model, "models/baseline_model.joblib")
        print("Model saved to models/baseline_model.joblib")
    else:
        print("Processed dataset not found. Run preprocessing first.")
