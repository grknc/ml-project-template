"""Tests for src/feature_engineering.py"""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from feature_engineering import build_preprocessor, save_preprocessor


@pytest.fixture
def numeric_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"n1": rng.standard_normal(20), "n2": rng.uniform(0, 1, 20)}
    )


@pytest.fixture
def categorical_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "c1": rng.choice(["a", "b"], 20),
            "c2": rng.choice(["x", "y", "z"], 20),
        }
    )


@pytest.fixture
def mixed_df(numeric_df, categorical_df):
    return pd.concat([numeric_df, categorical_df], axis=1)


class TestBuildPreprocessor:
    def test_returns_column_transformer(self, numeric_df):
        result = build_preprocessor(numeric_df)
        assert isinstance(result, ColumnTransformer)

    def test_numeric_columns_get_num_transformer(self, numeric_df):
        preprocessor = build_preprocessor(numeric_df)
        names = [name for name, _, _ in preprocessor.transformers]
        assert "num" in names

    def test_categorical_columns_get_cat_transformer(self, categorical_df):
        preprocessor = build_preprocessor(categorical_df)
        names = [name for name, _, _ in preprocessor.transformers]
        assert "cat" in names

    def test_mixed_df_has_both_transformers(self, mixed_df):
        preprocessor = build_preprocessor(mixed_df)
        names = [name for name, _, _ in preprocessor.transformers]
        assert "num" in names
        assert "cat" in names

    def test_bool_treated_as_numeric(self):
        df = pd.DataFrame({"flag": [True, False, True, False] * 5})
        preprocessor = build_preprocessor(df)
        names = [name for name, _, _ in preprocessor.transformers]
        assert "num" in names
        assert "cat" not in names

    def test_datetime_warns(self):
        df = pd.DataFrame(
            {
                "n": [1.0, 2.0],
                "dt": pd.to_datetime(["2021-01-01", "2021-02-01"]),
            }
        )
        with pytest.warns(UserWarning, match="Datetime"):
            build_preprocessor(df)

    def test_datetime_excluded_via_remainder_drop(self, numeric_df):
        preprocessor = build_preprocessor(numeric_df)
        assert preprocessor.remainder == "drop"

    def test_fit_transform_numeric(self, numeric_df):
        preprocessor = build_preprocessor(numeric_df)
        result = preprocessor.fit_transform(numeric_df)
        assert result.shape[0] == len(numeric_df)
        assert result.shape[1] == numeric_df.shape[1]

    def test_fit_transform_mixed(self, mixed_df):
        preprocessor = build_preprocessor(mixed_df)
        result = preprocessor.fit_transform(mixed_df)
        assert result.shape[0] == len(mixed_df)
        # OHE expands categorical columns → output cols >= input cols
        assert result.shape[1] >= mixed_df.shape[1]

    def test_handles_nulls_in_numeric(self):
        df = pd.DataFrame({"n": [1.0, float("nan"), 3.0, 4.0, 5.0]})
        preprocessor = build_preprocessor(df)
        result = preprocessor.fit_transform(df)
        assert not any(v != v for v in result.flatten())  # no NaN


class TestSavePreprocessor:
    def test_creates_file(self, tmp_path, numeric_df):
        preprocessor = build_preprocessor(numeric_df)
        preprocessor.fit(numeric_df)
        path = tmp_path / "preprocessor.joblib"
        save_preprocessor(preprocessor, path)
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path, numeric_df):
        preprocessor = build_preprocessor(numeric_df)
        preprocessor.fit(numeric_df)
        path = tmp_path / "models" / "sub" / "preprocessor.joblib"
        save_preprocessor(preprocessor, path)
        assert path.exists()

    def test_loaded_preprocessor_can_transform(self, tmp_path, numeric_df):
        preprocessor = build_preprocessor(numeric_df)
        preprocessor.fit(numeric_df)
        path = tmp_path / "preprocessor.joblib"
        save_preprocessor(preprocessor, path)
        loaded = joblib.load(path)
        result = loaded.transform(numeric_df)
        assert result.shape[0] == len(numeric_df)
