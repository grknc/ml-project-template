"""Data loading and preprocessing utilities.

This module is intentionally simple for bootcamp learners.
Fill in TODO sections based on your own dataset.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd


_SUPPORTED_FORMATS = {
    ".csv": pd.read_csv,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".json": pd.read_json,
    ".parquet": pd.read_parquet,
}


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load a dataset from the raw data directory.

    Supported formats: CSV, Excel (.xlsx/.xls), JSON, Parquet.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    reader = _SUPPORTED_FORMATS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(
            f"Unsupported file format '{path.suffix}'. "
            f"Supported: {list(_SUPPORTED_FORMATS)}"
        )

    return reader(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning steps.

    Notes for learners:
    - Replace these generic operations with domain-specific rules.
    - Keep transformations deterministic and documented.
    """
    clean_df = df.copy()
    original_shape = clean_df.shape

    # Remove duplicate rows.
    clean_df = clean_df.drop_duplicates()
    dropped_duplicates = original_shape[0] - len(clean_df)

    # Normalize column names.
    clean_df.columns = [col.strip().lower() for col in clean_df.columns]

    # Count nulls before filling.
    null_counts_before = clean_df.isnull().sum().sum()

    # Numeric columns: fill with median.
    for col in clean_df.select_dtypes(include=["number"]).columns:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    # Boolean columns: fill with mode.
    for col in clean_df.select_dtypes(include=["bool"]).columns:
        clean_df[col] = clean_df[col].fillna(clean_df[col].mode()[0])

    # Datetime columns: leave as-is, warn about nulls.
    datetime_cols = clean_df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    for col in datetime_cols:
        if clean_df[col].isnull().any():
            warnings.warn(
                f"Datetime column '{col}' has nulls — fill manually for your use case.",
                UserWarning,
                stacklevel=2,
            )

    # Remaining object/category columns: fill with "unknown".
    for col in clean_df.select_dtypes(include=["object", "category"]).columns:
        clean_df[col] = clean_df[col].fillna("unknown")

    null_counts_after = clean_df.isnull().sum().sum()

    print(
        f"[clean_data] {original_shape[0]} rows → {len(clean_df)} rows "
        f"({dropped_duplicates} duplicates removed) | "
        f"nulls filled: {null_counts_before - null_counts_after}"
    )

    return clean_df


def save_processed_data(df: pd.DataFrame, path: str | Path) -> None:
    """Save processed data to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":  # pragma: no cover
    raw_path = Path("data/raw/sample.csv")
    processed_path = Path("data/processed/sample_clean.csv")

    if not raw_path.exists():
        print("Place a sample dataset at data/raw/sample.csv to run this script.")
        sys.exit(1)

    data = load_raw_data(raw_path)
    clean = clean_data(data)
    save_processed_data(clean, processed_path)
    print(f"Saved processed data to: {processed_path}")
