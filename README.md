# Machine Learning Project Template & Industry-Based Examples

A complete starter template for Machine Learning bootcamp projects.

## Repository Purpose

This repository is both a **Machine Learning project template** and a **reference hub** with ~30 real-world, classical ML example projects for bootcamp participants. You can use the template to structure your own end-to-end workflow, and review the example projects to understand how different industries and problem types are handled in practice.

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
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
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
├── data/
│   ├── raw/                    # Original immutable source data
│   └── processed/              # Cleaned/transformed data for modeling
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling_evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
├── models/                     # Saved models (*.joblib, *.pkl)
└── reports/
    ├── figures/                # Plots and visual outputs
    └── metrics/                # Evaluation summaries (CSV/JSON/TXT)
```

---

## 3) Dataset Organization Guidelines

Use a consistent and reproducible data layout:

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
3. Clean + transform data.
4. Save processed output to `data/processed/`.

---

## :factory: Industry-Based Machine Learning Project Catalog

All example projects are located under `example_projects/`. Each project includes its own README with the problem definition, dataset details, modeling approach, and evaluation metrics. To make exploration easier, projects are grouped below by **industry** and **ML problem type**.

> Focus area: classical machine learning use cases (no deep learning and no NLP focus).

### A) Retail & E-commerce

#### Classification

- **Customer Satisfaction Prediction** (`example_projects/10_customer-satisfaction-prediction/`) - Predict whether customers are satisfied based on service, product, and transaction-related signals.
- **Marketing Campaign Response for E-commerce** (`example_projects/13_marketing-campaign-response-prediction/`) - Classify whether a customer is likely to respond to a product or campaign offer.

#### Regression

- **Sales Prediction** (`example_projects/05_sales-forecasting/`) - Estimate sales quantities or revenue values from historical and contextual variables.

#### Time Series & Forecasting

- **Store Item Demand Forecasting** (`example_projects/23_store-item-demand-forecasting/`) - Forecast future item-level demand per store from historical sales patterns.
- **Retail Sales Forecasting** (`example_projects/05_sales-forecasting/`) - Predict future sales trends over time for planning inventory and operations.

#### Unsupervised / Segmentation

- **Product Recommendation Analysis** (`example_projects/09_product-recommendation/`) - Use behavior patterns to group users/items and support recommendation logic.

### B) Marketing & CRM

#### Classification

- **Customer Churn Prediction** (`example_projects/01_customer-churn-prediction/`) - Predict which customers are likely to churn.
- **Marketing Campaign Response Prediction** (`example_projects/13_marketing-campaign-response-prediction/`) - Identify customers likely to convert in a campaign.

#### Regression

- **Customer Lifetime Value (CLTV) Prediction** (`example_projects/08_cltv-prediction/`) - Predict expected customer value over a future period.

#### Time Series & Forecasting

- **No dedicated time-series CRM project in the current set**.

#### Unsupervised / Segmentation

- **Customer Segmentation** (`example_projects/03_customer-segmentation/`) - Cluster customers into actionable segments for targeting and personalization.

### C) Finance & Insurance

#### Classification

- **Loan Default Prediction** (`example_projects/15_loan-default-prediction/`) - Predict whether a borrower will default.
- **Credit Risk Classification** (`example_projects/04_credit-risk-classification/`) - Classify applicants by credit risk category.
- **Fraud Detection** (`example_projects/07_fraud-detection/`) - Detect suspicious or fraudulent transactions.

#### Regression

- **Insurance Cost Prediction** (`example_projects/20_insurance-cost-prediction/`) - Estimate insurance charges from customer profile and risk factors.

#### Time Series & Forecasting

- **No dedicated finance time-series project in the current set**.

### D) Energy & Utilities

#### Classification

- **No dedicated classification project in the current set**.

#### Regression

- **Solar Power Generation Prediction** (`example_projects/21_solar-power-generation-prediction/`) - Predict generated solar power based on environmental and operational features.

#### Time Series & Forecasting

- **Energy Consumption Forecasting** (`example_projects/14_time-series-energy-consumption/`) - Forecast future energy usage from historical time-series data.
- **Solar Power Generation Forecasting** (`example_projects/21_solar-power-generation-prediction/`) - Forecast near-term power output for energy planning.

### E) Healthcare

#### Classification

- **Diabetes Prediction** (`example_projects/18_diabetes-prediction/`) - Classify whether a patient is likely to have diabetes.
- **Heart Disease Prediction** (`example_projects/19_heart-disease-prediction/`) - Predict heart disease risk from clinical indicators.

#### Regression

- **Calorie Intake Prediction** (`example_projects/16_calorie-intake-prediction/`) - Estimate daily calorie intake requirements from individual characteristics.

#### Time Series & Forecasting

- **No dedicated healthcare time-series project in the current set**.

### F) Gaming

#### Classification

- **User Retention / Churn-Like Behavior Prediction** (`example_projects/17_podcast-listening-prediction/`) - Classify whether users are likely to stay active or drop off based on behavior signals.

#### Regression

- **Player Engagement / Activity Prediction (closest behavior-based example)** (`example_projects/17_podcast-listening-prediction/`) - Predict continuous engagement levels such as listening/playtime-like behavior.

#### Time Series & Forecasting

- **No dedicated in-game activity forecasting project in the current set**.

### G) General / Cross-Industry

#### Classification

- **Binary Prediction with Rainfall Dataset** (`example_projects/25_binary-prediction-rainfall/`) - Predict a binary outcome from weather-related features.

#### Regression

- **California House Price Regression** (`example_projects/24_california-house-price-regression/`) - Predict house prices from demographic and location features.
- **House Price Prediction** (`example_projects/02_house-price-prediction/`) - Build a general-purpose regression model for real estate pricing.

#### Time Series & Forecasting

- **No dedicated cross-industry time-series project in the current set**.

#### Unsupervised / Segmentation

- **Employee Attrition Pattern Exploration** (`example_projects/06_employee-attrition-prediction/`) - Analyze workforce behavior patterns; can be adapted for segmentation and risk grouping.

#### Classification / Regression (Depending on Formulation)

- **Wine Quality Prediction** (`example_projects/22_wine-quality-prediction/`) - Model wine quality as a class label or a numeric score, depending on project framing.

---

## :compass: How Bootcamp Participants Should Use This Repository

- Fork or clone the repository to your own GitHub account.
- Choose **ONE industry** and **ONE problem type** (Classification, Regression, or Time Series & Forecasting).
- Use the main template structure (`data/`, `notebooks/`, `src/`, `models/`, `reports/`) for your own work.
- Do **NOT** submit an example project as your final bootcamp project.
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

## 4) Workflow Sections

### EDA

Goal: understand distributions, missingness, class balance, outliers, and relationships.

Checklist:

- Inspect shape, dtypes, null counts.
- Plot target distribution and key feature distributions.
- Note hypotheses and potential data quality issues.

### Feature Engineering

Goal: build model-ready inputs.

Checklist:

- Define numeric/categorical feature groups.
- Handle missing values.
- Encode categorical features.
- Scale or normalize numeric features where appropriate.

### Modeling

Goal: train baseline and improved models.

Checklist:

- Create train/validation (or train/test) split.
- Train at least one baseline model for your selected problem type.
- Track parameters and random seeds for reproducibility.

### Evaluation

Goal: quantify model quality and compare alternatives.

Checklist:

- Select proper metrics based on task type (classification/regression/forecasting).
- Visualize and compare model performance.
- Save metrics to `reports/metrics/`.

### Conclusion

Goal: communicate findings and next steps.

Checklist:

- Summarize best-performing model.
- Highlight limitations and assumptions.
- Suggest concrete improvements and deployment considerations.

---

## 5) Example Snippets

### Loading and Cleaning Data

```python
import pandas as pd

# Load raw dataset
df = pd.read_csv("data/raw/your_dataset.csv")

# Basic cleaning
# TODO: customize for your dataset

df = df.drop_duplicates()
df.columns = [c.strip().lower() for c in df.columns]

# Example missing-value handling
for col in df.select_dtypes(include=["number"]).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(exclude=["number"]).columns:
    df[col] = df[col].fillna("unknown")

# Save processed file
df.to_csv("data/processed/clean_dataset.csv", index=False)
```

### Training a Classification Model

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# TODO: replace 'target' with your target column name
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

### Evaluation with Accuracy and ROC-AUC

```python
from sklearn.metrics import accuracy_score, roc_auc_score

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]  # for binary classification

accuracy = accuracy_score(y_test, preds)
roc_auc = roc_auc_score(y_test, probs)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
```

### Saving a Trained Model (joblib or pickle)

```python
import joblib
import pickle

# Option 1: joblib (recommended for many sklearn objects)
joblib.dump(model, "models/baseline_model.joblib")

# Option 2: pickle
with open("models/baseline_model.pkl", "wb") as f:
    pickle.dump(model, f)
```

---

## 6) Suggested Next Steps for Learners

- Add project-specific preprocessing and feature pipelines in `src/`.
- Track experiments (metrics, parameters, plots) under `reports/`.
- Refactor notebook code into reusable functions/modules.
- Add tests and CI once your project scope is stable.

Happy building 🚀
