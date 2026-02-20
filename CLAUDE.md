# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A starter template for classical ML bootcamp projects. Learners fork this repo, place their dataset in `data/raw/`, and work through notebooks before refactoring logic into `src/` modules.

## Python Environment

The working Python is the Anaconda base installation:
```bash
/c/Users/proje/anaconda3/python.exe
```
Use this explicitly when running scripts or tests (standard `python` / `python3` commands resolve to a stub on this machine).

## Setup

```bash
/c/Users/proje/anaconda3/python.exe -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Pipeline

```bash
python src/data_preprocessing.py   # raw → data/processed/sample_clean.csv
python src/train.py                # processed CSV → models/baseline_model.joblib
                                   #               + models/preprocessor.joblib
                                   #               + reports/metrics/baseline_metrics.json
python src/evaluate.py             # reload model → print + save metrics
jupyter notebook
```

## Running Tests

```bash
/c/Users/proje/anaconda3/python.exe -m pytest tests/
/c/Users/proje/anaconda3/python.exe -m pytest tests/test_train.py        # single file
/c/Users/proje/anaconda3/python.exe -m pytest tests/ -k "TestTrainModel" # single class
```

Coverage is configured in `pytest.ini` (`--cov=src --cov-report=term-missing`). All `__main__` blocks carry `# pragma: no cover`.

## Architecture

Linear pipeline across four modules in `src/`:

1. **`data_preprocessing.py`** — `load_raw_data(path)` supports CSV, JSON, Excel, Parquet (dispatches via `_SUPPORTED_FORMATS` dict). `clean_data(df)` deduplicates, normalises column names, and fills nulls per dtype: numeric→median, bool→mode, datetime→warn+skip, object/category→`"unknown"`. Prints a summary line. `save_processed_data(df, path)` creates parent dirs automatically.

2. **`feature_engineering.py`** — `build_preprocessor(X: pd.DataFrame)` takes the **feature matrix only** (target already dropped). Detects numeric+bool (→ median impute + StandardScaler) and object/category (→ mode impute + OneHotEncoder) columns; warns and drops datetime columns via `remainder="drop"`. Returns an unfitted `ColumnTransformer`. `save_preprocessor(preprocessor, path)` persists the fitted transformer via joblib.

3. **`train.py`** — `train_model(df, target_col)` splits X/y, calls `build_preprocessor(X)`, wraps it with `LogisticRegression` in a `Pipeline`, does 80/20 stratified split, fits, and returns `(model, X_test, y_test)`. `__main__` also calls `evaluate_classification` and saves both `models/baseline_model.joblib` and `models/preprocessor.joblib`.

4. **`evaluate.py`** — `evaluate_classification(model, X_test, y_test)` returns `{"accuracy", "roc_auc", "classification_report"}`. ROC-AUC is binary (`proba[:, 1]`) or multiclass (`ovr` macro) depending on `y_test.nunique()`. Issues `UserWarning` when model lacks `predict_proba`. `save_metrics` writes JSON.

**Key coupling:** `train.py` imports `build_preprocessor` and `save_preprocessor` from `feature_engineering.py` at module level (with a `sys.path.insert` guard so the file is runnable from any CWD).

Notebooks in `notebooks/` mirror the pipeline: `01_eda` → `02_feature_engineering` → `03_modeling_evaluation`. All three read from `data/processed/sample_clean.csv`; they do **not** pass data between each other.

## Example Projects

`example_projects/` contains 25 reference projects, each with a `README.md` and a `starter_notebook.ipynb`. Projects are grouped by industry (Retail, Finance, Healthcare, Media, etc.) and cover 7 problem types: binary classification (standard + imbalanced), regression, time series, clustering, collaborative filtering, text classification.

To regenerate all starter notebooks after template changes:
```bash
python generate_notebooks.py
```
