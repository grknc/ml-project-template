"""Data loading and preprocessing utilities.

This module is intentionally simple for bootcamp learners.
Fill in TODO sections based on your own dataset.
"""

from pathlib import Path

import pandas as pd


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset from the raw data directory."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning steps.

    Notes for learners:
    - Replace these generic operations with domain-specific rules.
    - Keep transformations deterministic and documented.
    """
    clean_df = df.copy()

    # Remove duplicate rows.
    clean_df = clean_df.drop_duplicates()

    # Normalize column names.
    clean_df.columns = [col.strip().lower() for col in clean_df.columns]

    # Fill missing values with simple defaults.
    for col in clean_df.select_dtypes(include=["number"]).columns:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    for col in clean_df.select_dtypes(exclude=["number"]).columns:
        clean_df[col] = clean_df[col].fillna("unknown")

    return clean_df


def save_processed_data(df: pd.DataFrame, path: str | Path) -> None:
    """Save processed data to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    # Example local run:
    # python src/data_preprocessing.py
    raw_path = Path("data/raw/sample.csv")
    processed_path = Path("data/processed/sample_clean.csv")

    if raw_path.exists():
        data = load_raw_data(raw_path)
        clean = clean_data(data)
        save_processed_data(clean, processed_path)
        print(f"Saved processed data to: {processed_path}")
    else:
        print("Place a sample dataset at data/raw/sample.csv to run this script.")
