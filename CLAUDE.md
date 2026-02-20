# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A starter template for classical ML bootcamp projects. Learners fork this repo, place their dataset in `data/raw/`, and work through notebooks before refactoring logic into `src/` modules.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Pipeline

Each `src/` module is independently executable:

```bash
python src/data_preprocessing.py   # raw → processed CSV
python src/train.py                # processed CSV → saved model in models/
python src/evaluate.py             # model → metrics JSON in reports/metrics/
```

Start Jupyter for notebook-based workflow:

```bash
jupyter notebook
```

## Architecture

The project follows a linear pipeline across four modules:

1. **`src/data_preprocessing.py`** — Loads raw CSV, deduplicates, normalizes column names, fills missing values, saves to `data/processed/`.
2. **`src/feature_engineering.py`** — Builds a sklearn `ColumnTransformer` with numeric (median impute + StandardScaler) and categorical (mode impute + OneHotEncoder) pipelines via `build_preprocessor(df, target_col)`.
3. **`src/train.py`** — Wraps the preprocessor + LogisticRegression in a Pipeline, does 80/20 split, saves model via joblib to `models/`.
4. **`src/evaluate.py`** — Loads model, computes accuracy and ROC-AUC, saves metrics JSON to `reports/metrics/`.

Notebooks in `notebooks/` mirror this sequence: `01_eda` → `02_feature_engineering` → `03_modeling_evaluation`.

## Data Conventions

- `data/raw/` — original, immutable source files
- `data/processed/` — cleaned output from `data_preprocessing.py`
- `models/` — joblib-serialized model artifacts
- `reports/metrics/` — JSON/CSV evaluation outputs
- `reports/figures/` — saved plots

## Example Projects

`example_projects/` contains 25 reference projects organized by industry (Retail, Finance, Healthcare, etc.) and problem type (classification, regression, time series, clustering). Each has its own README with dataset source, recommended models, and evaluation metrics.
