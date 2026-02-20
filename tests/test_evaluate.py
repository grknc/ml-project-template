"""Tests for src/evaluate.py"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import evaluate_classification, save_metrics


class _NoProbModel:
    """Minimal classifier without predict_proba, for testing the warning path."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


@pytest.fixture
def binary_model_and_data():
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.uniform(0, 1, n)})
    y = pd.Series([0] * 50 + [1] * 50)
    model = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]
    )
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def multiclass_model_and_data():
    rng = np.random.default_rng(1)
    n = 90
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.uniform(0, 1, n)})
    y = pd.Series([0] * 30 + [1] * 30 + [2] * 30)
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    return model, X, y


class TestEvaluateClassification:
    def test_returns_accuracy(self, binary_model_and_data):
        model, X, y = binary_model_and_data
        metrics = evaluate_classification(model, X, y)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_returns_roc_auc_for_binary(self, binary_model_and_data):
        model, X, y = binary_model_and_data
        metrics = evaluate_classification(model, X, y)
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] is not None
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_returns_classification_report_dict(self, binary_model_and_data):
        model, X, y = binary_model_and_data
        metrics = evaluate_classification(model, X, y)
        assert "classification_report" in metrics
        assert isinstance(metrics["classification_report"], dict)

    def test_roc_auc_multiclass_ovr(self, multiclass_model_and_data):
        model, X, y = multiclass_model_and_data
        metrics = evaluate_classification(model, X, y)
        assert metrics["roc_auc"] is not None
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_no_predict_proba_roc_auc_is_none(self):
        rng = np.random.default_rng(2)
        X = pd.DataFrame({"a": rng.standard_normal(20)})
        y = pd.Series([0] * 10 + [1] * 10)
        model = _NoProbModel().fit(X, y)
        with pytest.warns(UserWarning, match="predict_proba"):
            metrics = evaluate_classification(model, X, y)
        assert metrics["roc_auc"] is None

    def test_accuracy_matches_manual(self, binary_model_and_data):
        from sklearn.metrics import accuracy_score

        model, X, y = binary_model_and_data
        metrics = evaluate_classification(model, X, y)
        expected = accuracy_score(y, model.predict(X))
        assert metrics["accuracy"] == pytest.approx(expected)


class TestSaveMetrics:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "metrics.json"
        save_metrics({"accuracy": 0.9, "roc_auc": 0.85, "classification_report": {}}, path)
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "reports" / "metrics" / "m.json"
        save_metrics({"accuracy": 0.9, "roc_auc": 0.85, "classification_report": {}}, path)
        assert path.exists()

    def test_json_content_accurate(self, tmp_path):
        path = tmp_path / "metrics.json"
        save_metrics({"accuracy": 0.92, "roc_auc": 0.88, "classification_report": {}}, path)
        loaded = json.loads(path.read_text())
        assert loaded["accuracy"] == pytest.approx(0.92)
        assert loaded["roc_auc"] == pytest.approx(0.88)

    def test_full_metrics_dict_serializable(self, tmp_path, binary_model_and_data):
        model, X, y = binary_model_and_data
        metrics = evaluate_classification(model, X, y)
        path = tmp_path / "metrics.json"
        save_metrics(metrics, path)  # should not raise
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert "accuracy" in loaded
        assert "classification_report" in loaded
