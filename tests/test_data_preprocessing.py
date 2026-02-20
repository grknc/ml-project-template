"""Tests for src/data_preprocessing.py"""

import warnings

import pandas as pd
import pytest

from data_preprocessing import clean_data, load_raw_data, save_processed_data


class TestLoadRawData:
    def test_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)
        df = load_raw_data(path)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["a", "b"]

    def test_json(self, tmp_path):
        path = tmp_path / "data.json"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_json(path)
        df = load_raw_data(path)
        assert df.shape == (2, 2)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_raw_data("no/such/file.csv")

    def test_unsupported_format(self, tmp_path):
        bad = tmp_path / "data.txt"
        bad.write_text("a,b\n1,2")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_raw_data(bad)

    def test_unsupported_format_message_lists_supported(self, tmp_path):
        bad = tmp_path / "data.xml"
        bad.write_text("<root/>")
        with pytest.raises(ValueError, match=r"\.csv"):
            load_raw_data(bad)


class TestCleanData:
    def test_removes_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        result = clean_data(df)
        assert len(result) == 2

    def test_normalizes_column_names(self):
        df = pd.DataFrame({"  A ": [1], " B_Col ": [2]})
        result = clean_data(df)
        assert list(result.columns) == ["a", "b_col"]

    def test_fills_numeric_nulls_with_median(self):
        df = pd.DataFrame({"a": [1.0, float("nan"), 3.0]})
        result = clean_data(df)
        assert result["a"].isnull().sum() == 0
        assert result["a"].iloc[1] == pytest.approx(2.0)

    def test_fills_object_nulls_with_unknown(self):
        df = pd.DataFrame({"cat": ["x", None, "y"]})
        result = clean_data(df)
        assert result["cat"].isnull().sum() == 0
        assert result["cat"].iloc[1] == "unknown"

    def test_bool_columns_pass_through(self):
        df = pd.DataFrame({"flag": [True, False, True, False]})
        result = clean_data(df)
        assert "flag" in result.columns
        assert result["flag"].isnull().sum() == 0

    def test_warns_on_datetime_nulls(self):
        df = pd.DataFrame(
            {"dt": pd.to_datetime(["2021-01-01", None, "2021-03-01"])}
        )
        with pytest.warns(UserWarning, match="Datetime column"):
            clean_data(df)

    def test_no_userwarning_when_datetime_has_no_nulls(self):
        df = pd.DataFrame(
            {"dt": pd.to_datetime(["2021-01-01", "2021-02-01"])}
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clean_data(df)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_returns_copy(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = clean_data(df)
        assert result is not df

    def test_prints_summary(self, capsys):
        df = pd.DataFrame({"a": [1, 1, 2]})
        clean_data(df)
        captured = capsys.readouterr()
        assert "[clean_data]" in captured.out


class TestSaveProcessedData:
    def test_creates_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "processed.csv"
        save_processed_data(df, path)
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        path = tmp_path / "nested" / "dir" / "data.csv"
        save_processed_data(df, path)
        assert path.exists()

    def test_saved_without_index(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        path = tmp_path / "data.csv"
        save_processed_data(df, path)
        loaded = pd.read_csv(path)
        assert "Unnamed: 0" not in loaded.columns
        assert list(loaded.columns) == ["a"]

    def test_roundtrip(self, tmp_path):
        df = pd.DataFrame({"x": [10, 20, 30], "y": ["a", "b", "c"]})
        path = tmp_path / "data.csv"
        save_processed_data(df, path)
        loaded = pd.read_csv(path)
        assert loaded.shape == df.shape
