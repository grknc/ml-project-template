# Machine Learning Project Template & Industry-Based Examples

A complete starter template for Machine Learning bootcamp projects.

## Repository Purpose

This repository is both a **Machine Learning project template** and a **reference hub** with 25 real-world, classical ML example projects for bootcamp participants. Use the template to structure your own end-to-end workflow, and review the example projects to understand how different industries and problem types are handled in practice.

---

## 1) How Learners Should Fork and Start

1. Click **Fork** (top-right on GitHub) to create your own copy.
2. Clone your fork locally:

   ```bash
   git clone https://github.com/<your-username>/ml-project-template.git
   cd ml-project-template
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start working through notebooks in order:
   - `notebooks/01_eda.ipynb`
   - `notebooks/02_feature_engineering.ipynb`
   - `notebooks/03_modeling_evaluation.ipynb`

---

## 2) Folder Structure

```text
ml-project-template/
├── README.md
├── requirements.txt
├── pytest.ini                  # Test configuration
├── data/
│   ├── raw/                    # Original immutable source data
│   └── processed/              # Cleaned/transformed data for modeling
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling_evaluation.ipynb
├── src/
│   ├── data_preprocessing.py   # load → clean → save pipeline
│   ├── feature_engineering.py  # ColumnTransformer builder
│   ├── train.py                # model training & saving
│   └── evaluate.py             # metrics & reporting
├── tests/
│   ├── conftest.py
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_train.py
│   └── test_evaluate.py
├── example_projects/           # 25 industry-based ML examples
│   └── <project>/
│       ├── README.md
│       └── starter_notebook.ipynb
├── models/                     # Saved models (*.joblib)
└── reports/
    ├── figures/                # Plots and visual outputs
    └── metrics/                # Evaluation summaries (JSON/CSV)
```

---

## 3) Dataset Organization Guidelines

- Put untouched source files in `data/raw/`.
- Write cleaned or engineered datasets to `data/processed/`.
- Never overwrite raw data.
- Prefer meaningful names, e.g.:
  - `data/raw/customer_churn.csv`
  - `data/processed/customer_churn_clean.csv`
- If data is sensitive, do not commit it to Git. Add patterns to `.gitignore` as needed.

Recommended workflow:

1. Load from `data/raw/`.
2. Validate schema and missing values.
3. Clean + transform → save to `data/processed/`.
4. Train and evaluate from `data/processed/`.

---

## 4) Running the Pipeline

Each `src/` module can be run standalone in sequence:

```bash
python src/data_preprocessing.py   # raw → data/processed/sample_clean.csv
python src/train.py                # processed CSV → models/baseline_model.joblib
                                   #               + models/preprocessor.joblib
                                   #               + reports/metrics/baseline_metrics.json
python src/evaluate.py             # reload model → print metrics
```

---

## 5) Running Tests

```bash
python -m pytest tests/
```

The suite covers all `src/` modules with 100% coverage (excluding CLI `__main__` blocks).

---

## :factory: Industry-Based Machine Learning Project Catalog

All 25 example projects are in `example_projects/`. Each project includes a `README.md` (problem definition, dataset, recommended models, metrics) and a `starter_notebook.ipynb` ready to run after downloading the dataset.

### A) Retail & E-commerce

| # | Project | Type |
|---|---|---|
| 10 | [Customer Satisfaction Prediction](example_projects/10_customer-satisfaction-prediction/) | Classification |
| 05 | [Sales Forecasting (Walmart)](example_projects/05_sales-forecasting/) | Time Series |
| 23 | [Store Item Demand Forecasting](example_projects/23_store-item-demand-forecasting/) | Time Series |
| 09 | [Product Recommendation (MovieLens)](example_projects/09_product-recommendation/) | Recommender System |

### B) Marketing & CRM

| # | Project | Type |
|---|---|---|
| 01 | [Customer Churn Prediction](example_projects/01_customer-churn-prediction/) | Classification |
| 13 | [Marketing Campaign Response](example_projects/13_marketing-campaign-response-prediction/) | Classification |
| 08 | [Customer Lifetime Value (CLTV)](example_projects/08_cltv-prediction/) | Regression |
| 03 | [Customer Segmentation](example_projects/03_customer-segmentation/) | Clustering |

### C) Finance & Insurance

| # | Project | Type |
|---|---|---|
| 04 | [Credit Risk Classification](example_projects/04_credit-risk-classification/) | Classification |
| 07 | [Credit Card Fraud Detection](example_projects/07_fraud-detection/) | Classification (imbalanced) |
| 15 | [Loan Default Prediction](example_projects/15_loan-default-prediction/) | Classification |
| 20 | [Insurance Cost Prediction](example_projects/20_insurance-cost-prediction/) | Regression |

### D) Energy & Utilities

| # | Project | Type |
|---|---|---|
| 14 | [Energy Consumption Forecasting](example_projects/14_time-series-energy-consumption/) | Time Series |
| 21 | [Solar Power Generation Forecasting](example_projects/21_solar-power-generation-prediction/) | Time Series |

### E) Healthcare

| # | Project | Type |
|---|---|---|
| 18 | [Diabetes Risk Prediction](example_projects/18_diabetes-prediction/) | Classification |
| 19 | [Heart Disease Prediction](example_projects/19_heart-disease-prediction/) | Classification |
| 16 | [Calorie Expenditure Prediction](example_projects/16_calorie-intake-prediction/) | Regression |

### F) HR & Workforce

| # | Project | Type |
|---|---|---|
| 06 | [Employee Attrition Prediction](example_projects/06_employee-attrition-prediction/) | Classification |

### G) Media & Entertainment

| # | Project | Type |
|---|---|---|
| 11 | [Movie Review Sentiment Analysis](example_projects/11_movie-review-sentiment-analysis/) | Text Classification |
| 12 | [Email / SMS Spam Detection](example_projects/12_email-spam-classification/) | Text Classification |
| 17 | [Podcast Listening Time Prediction](example_projects/17_podcast-listening-prediction/) | Regression |

### H) General / Cross-Industry

| # | Project | Type |
|---|---|---|
| 02 | [House Price Prediction](example_projects/02_house-price-prediction/) | Regression |
| 24 | [California Housing Price Regression](example_projects/24_california-house-price-regression/) | Regression |
| 22 | [Wine Quality Prediction](example_projects/22_wine-quality-prediction/) | Classification |
| 25 | [Binary Rainfall Prediction](example_projects/25_binary-prediction-rainfall/) | Classification |

---

## :compass: How Bootcamp Participants Should Use This Repository

- Fork or clone the repository to your own GitHub account.
- Choose **ONE industry** and **ONE problem type** (Classification, Regression, or Time Series).
- Use the main template structure (`data/`, `notebooks/`, `src/`, `models/`, `reports/`) for your own work.
- Use `example_projects/<project>/starter_notebook.ipynb` as reference — do **not** submit an example project as your final work.
- Build your own dataset-driven solution and document your decisions clearly.

---

## :white_check_mark: Expectations for Bootcamp Projects

Your final project should include:

- A clear problem definition
- Exploratory Data Analysis (EDA)
- Feature engineering
- At least two models for comparison
- Proper evaluation metrics aligned with the problem type
- Business / real-world interpretation of the results

---

## 6) Workflow Sections

### EDA

Goal: understand distributions, missingness, class balance, outliers, and relationships.

Checklist:

- Inspect shape, dtypes, null counts.
- Plot target distribution and key feature distributions.
- Note hypotheses and potential data quality issues.

### Feature Engineering

Goal: build model-ready inputs.

Checklist:

- Define numeric/categorical/boolean feature groups.
- Handle missing values (median for numeric, mode for categorical).
- Encode categorical features with `OneHotEncoder`.
- Scale numeric features where appropriate.

### Modeling

Goal: train baseline and improved models.

Checklist:

- Create train/test split (`stratify=y` for classification).
- Wrap preprocessing + model in a `sklearn.Pipeline`.
- Track parameters and random seeds for reproducibility.

### Evaluation

Goal: quantify model quality and compare alternatives.

Checklist:

- Select proper metrics: ROC-AUC / F1 (classification), MAE / RMSE / R² (regression), MAPE (time series).
- For imbalanced datasets, prefer Recall, F1, and PR-AUC over Accuracy.
- Save metrics to `reports/metrics/`.

### Conclusion

Goal: communicate findings and next steps.

Checklist:

- Summarize best-performing model and key metric.
- Highlight limitations and assumptions.
- Suggest concrete improvements (hyperparameter tuning, more data, feature ideas).

---

## 7) Example Snippets

### Loading and Cleaning Data

```python
# Using the src/ module directly:
from src.data_preprocessing import load_raw_data, clean_data, save_processed_data

df = load_raw_data("data/raw/your_dataset.csv")  # also supports .xlsx, .json, .parquet
df = clean_data(df)
save_processed_data(df, "data/processed/clean_dataset.csv")
```

### Training a Classification Model

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Import src/ module
import sys; sys.path.insert(0, "src")
from feature_engineering import build_preprocessor

df = pd.read_csv("data/processed/clean_dataset.csv")
TARGET = "target"  # replace with your column name

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", build_preprocessor(X_train)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)
```

### Evaluation

```python
from sklearn.metrics import classification_report, roc_auc_score

preds = model.predict(X_test)
probs = model.predict_proba(X_test)

# Binary
roc_auc = roc_auc_score(y_test, probs[:, 1])

# Multiclass
# roc_auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")

print(f"ROC-AUC: {roc_auc:.4f}")
print(classification_report(y_test, preds))
```

### Saving a Trained Model

```python
import joblib

joblib.dump(model, "models/baseline_model.joblib")

# Reload later:
# loaded = joblib.load("models/baseline_model.joblib")
```

---

## 8) Suggested Next Steps for Learners

- Add project-specific preprocessing and feature pipelines in `src/`.
- Track experiments (metrics, parameters, plots) under `reports/`.
- Refactor notebook code into reusable functions/modules in `src/`.
- Write tests for your modules using `pytest`.

Happy building 🚀
