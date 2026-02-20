"""Feature engineering utilities for tabular ML tasks."""

import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a sklearn preprocessor for numeric, boolean, and categorical features.

    Accepts the feature matrix X (target column must be dropped before calling).
    Learners can extend this by:
    - changing imputation strategies,
    - adding custom transformers,
    - applying log transforms or feature interactions.
    """
    # Datetime columns cannot be processed by sklearn — warn and exclude via remainder="drop".
    datetime_cols = X.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    if datetime_cols:
        warnings.warn(
            f"Datetime columns will be dropped (not yet supported): {datetime_cols}",
            UserWarning,
            stacklevel=2,
        )

    # bool is a numeric subtype; treat alongside numeric features.
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # drops datetime and any other unsupported column types
    )


def save_preprocessor(
    preprocessor, output_path: str | Path = "models/preprocessor.joblib"
) -> None:
    """Save a fitted preprocessor to disk for use during inference."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_path)