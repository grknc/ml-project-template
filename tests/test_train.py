"""Tests for src/train.py"""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from train import save_model, train_model


class TestTrainModel:
    def test_raises_if_target_missing(self, binary_df):
        with pytest.raises(ValueError, match="Target column"):
            train_model(binary_df, target_col="nonexistent")

    def test_returns_three_values(self, binary_df):
        result = train_model(binary_df)
        assert len(result) == 3

    def test_returns_fitted_pipeline(self, binary_df):
        model, _, _ = train_model(binary_df)
        assert isinstance(model, Pipeline)

    def test_pipeline_has_preprocessor_step(self, binary_df):
        model, _, _ = train_model(binary_df)
        assert "preprocessor" in model.named_steps

    def test_pipeline_has_classifier_step(self, binary_df):
        model, _, _ = train_model(binary_df)
        assert "classifier" in model.named_steps

    def test_test_split_is_20_percent(self, binary_df):
        _, X_test, y_test = train_model(binary_df)
        expected = int(len(binary_df) * 0.2)
        assert len(X_test) == expected
        assert len(y_test) == expected

    def test_model_can_predict(self, binary_df):
        model, X_test, _ = train_model(binary_df)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_predictions_are_binary(self, binary_df):
        model, X_test, _ = train_model(binary_df)
        preds = model.predict(X_test)
        assert set(preds).issubset({0, 1})

    def test_custom_target_col(self):
        rng = np.random.default_rng(7)
        n = 40
        df = pd.DataFrame(
            {
                "feat": rng.standard_normal(n),
                "label": [0] * 20 + [1] * 20,
            }
        )
        model, X_test, y_test = train_model(df, target_col="label")
        assert len(X_test) == int(n * 0.2)


class TestSaveModel:
    def test_creates_file(self, tmp_path, binary_df):
        model, _, _ = train_model(binary_df)
        path = tmp_path / "model.joblib"
        save_model(model, path)
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path, binary_df):
        model, _, _ = train_model(binary_df)
        path = tmp_path / "models" / "baseline.joblib"
        save_model(model, path)
        assert path.exists()

    def test_saved_model_loadable_and_predicts(self, tmp_path, binary_df):
        model, X_test, _ = train_model(binary_df)
        path = tmp_path / "model.joblib"
        save_model(model, path)
        loaded = joblib.load(path)
        preds = loaded.predict(X_test)
        assert len(preds) == len(X_test)
