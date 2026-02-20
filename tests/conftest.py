"""Shared pytest fixtures and path setup."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src/ importable from tests without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def binary_df():
    """Balanced binary classification DataFrame, no nulls."""
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame(
        {
            "num1": rng.standard_normal(n),
            "num2": rng.uniform(0, 10, n),
            "cat1": rng.choice(["a", "b", "c"], n),
            "target": [0] * 30 + [1] * 30,
        }
    )


@pytest.fixture
def feature_df(binary_df):
    """Feature matrix (no target column)."""
    return binary_df.drop(columns=["target"])
