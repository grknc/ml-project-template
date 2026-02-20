"""One-off script: generates starter_notebook.ipynb for all 25 example projects."""

import json
from pathlib import Path

BASE = Path(__file__).parent / "example_projects"


# ── project metadata ──────────────────────────────────────────────────────────
PROJECTS = {
    "01_customer-churn-prediction": {
        "title": "Bank Customer Churn Prediction",
        "url": "https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data",
        "file": "bank_customer_churn.csv",
        "target": "churn",
        "type": "clf_imbalanced",
        "desc": "Predict whether a bank customer will leave based on their profile and account activity.",
    },
    "02_house-price-prediction": {
        "title": "House Prices: Advanced Regression",
        "url": "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques",
        "file": "train.csv",
        "target": "SalePrice",
        "type": "regression",
        "desc": "Predict house sale prices from property features.",
    },
    "03_customer-segmentation": {
        "title": "Customer Segmentation with Clustering",
        "url": "https://www.kaggle.com/datasets/yasserh/customer-segmentation-dataset/data",
        "file": "customers.csv",
        "target": None,
        "type": "clustering",
        "desc": "Segment customers into meaningful groups using unsupervised clustering.",
    },
    "04_credit-risk-classification": {
        "title": "Credit Risk Classification",
        "url": "https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data",
        "file": "credit_risk_dataset.csv",
        "target": "loan_status",
        "type": "clf_imbalanced",
        "desc": "Classify loan applicants by credit risk level.",
    },
    "05_sales-forecasting": {
        "title": "Walmart Sales Forecasting",
        "url": "https://www.kaggle.com/datasets/yasserh/walmart-dataset/data",
        "file": "Walmart.csv",
        "target": "Weekly_Sales",
        "type": "timeseries",
        "date_col": "Date",
        "desc": "Forecast weekly store sales using historical Walmart data.",
    },
    "06_employee-attrition-prediction": {
        "title": "HR Analytics: Employee Attrition Prediction",
        "url": "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset",
        "file": "WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "target": "Attrition",
        "type": "clf_imbalanced",
        "desc": "Predict whether an employee will leave the company.",
    },
    "07_fraud-detection": {
        "title": "Credit Card Fraud Detection",
        "url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "file": "creditcard.csv",
        "target": "Class",
        "type": "clf_imbalanced",
        "desc": "Detect fraudulent credit card transactions in a highly imbalanced dataset.",
    },
    "08_cltv-prediction": {
        "title": "Customer Lifetime Value Prediction",
        "url": "https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci",
        "file": "online_retail_II.csv",
        "target": "TotalRevenue",
        "type": "regression",
        "desc": "Predict future customer value from transaction history using RFM features.",
    },
    "09_product-recommendation": {
        "title": "MovieLens Recommendation System",
        "url": "https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset",
        "file": "ratings.csv",
        "target": "rating",
        "type": "recommender",
        "desc": "Build a collaborative filtering recommendation system for movies.",
        "user_col": "userId",
        "item_col": "movieId",
    },
    "10_customer-satisfaction-prediction": {
        "title": "Airline Passenger Satisfaction",
        "url": "https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction",
        "file": "train.csv",
        "target": "satisfaction",
        "type": "clf_standard",
        "desc": "Predict whether an airline passenger is satisfied based on service ratings.",
    },
    "11_movie-review-sentiment-analysis": {
        "title": "Movie Review Sentiment Classification",
        "url": "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
        "file": "IMDB Dataset.csv",
        "target": "sentiment",
        "text_col": "review",
        "type": "text_clf",
        "desc": "Classify movie reviews as positive or negative using TF-IDF.",
    },
    "12_email-spam-classification": {
        "title": "Email / SMS Spam Detection",
        "url": "https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset",
        "file": "spam.csv",
        "target": "v1",
        "text_col": "v2",
        "type": "text_clf",
        "desc": "Classify messages as spam or ham using TF-IDF and text classifiers.",
    },
    "13_marketing-campaign-response-prediction": {
        "title": "Marketing Campaign Response Prediction",
        "url": "https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset",
        "file": "bank.csv",
        "target": "deposit",
        "type": "clf_standard",
        "desc": "Predict which customers will respond to a marketing campaign.",
    },
    "14_time-series-energy-consumption": {
        "title": "Energy Consumption Forecasting",
        "url": "https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set",
        "file": "household_power_consumption.txt",
        "target": "Global_active_power",
        "type": "timeseries",
        "date_col": "Date",
        "desc": "Forecast household energy consumption using time series features.",
    },
    "15_loan-default-prediction": {
        "title": "Loan Default Risk Classification",
        "url": "https://www.kaggle.com/datasets/wordsforthewise/lending-club",
        "file": "accepted_2007_to_2018Q4.csv",
        "target": "loan_status",
        "type": "clf_imbalanced",
        "desc": "Classify loan applications by likelihood of default.",
    },
    "16_calorie-intake-prediction": {
        "title": "Calorie Expenditure Prediction",
        "url": "https://www.kaggle.com/datasets/brsdincer/calorie-expenditure-exercise-dataset",
        "file": "calories.csv",
        "target": "Calories",
        "type": "regression",
        "desc": "Predict calorie expenditure from personal and activity features.",
    },
    "17_podcast-listening-prediction": {
        "title": "Podcast Listening Time Prediction",
        "url": "https://www.kaggle.com/competitions/playground-series-s4e7",
        "file": "train.csv",
        "target": "listening_time",
        "type": "regression",
        "desc": "Predict podcast listening duration from user behavior signals.",
    },
    "18_diabetes-prediction": {
        "title": "Diabetes Risk Prediction",
        "url": "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database",
        "file": "diabetes.csv",
        "target": "Outcome",
        "type": "clf_imbalanced",
        "desc": "Predict diabetes risk from medical measurements.",
    },
    "19_heart-disease-prediction": {
        "title": "Heart Disease Risk Classification",
        "url": "https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction",
        "file": "heart.csv",
        "target": "HeartDisease",
        "type": "clf_imbalanced",
        "desc": "Classify patients by heart disease risk using clinical indicators.",
    },
    "20_insurance-cost-prediction": {
        "title": "Insurance Premium Cost Prediction",
        "url": "https://www.kaggle.com/datasets/mirichoi0218/insurance",
        "file": "insurance.csv",
        "target": "charges",
        "type": "regression",
        "desc": "Predict medical insurance charges from age, BMI, smoking status, and region.",
    },
    "21_solar-power-generation-prediction": {
        "title": "Solar Power Generation Forecasting",
        "url": "https://www.kaggle.com/datasets/anikannal/solar-power-generation-data/data",
        "file": "Plant_1_Generation_Data.csv",
        "target": "DC_POWER",
        "type": "timeseries",
        "date_col": "DATE_TIME",
        "desc": "Forecast solar power output using weather and irradiation features.",
    },
    "22_wine-quality-prediction": {
        "title": "Wine Quality Prediction",
        "url": "https://www.kaggle.com/datasets/yasserh/wine-quality-dataset",
        "file": "WineQT.csv",
        "target": "quality",
        "type": "clf_standard",
        "desc": "Predict wine quality class from physicochemical properties.",
    },
    "23_store-item-demand-forecasting": {
        "title": "Store Item Demand Forecasting",
        "url": "https://www.kaggle.com/competitions/demand-forecasting-kernels-only",
        "file": "train.csv",
        "target": "sales",
        "type": "timeseries",
        "date_col": "date",
        "desc": "Forecast store-item demand using lag features and seasonality.",
    },
    "24_california-house-price-regression": {
        "title": "California Housing Price Regression",
        "url": "https://www.kaggle.com/competitions/regression-tabular-california-housing",
        "file": "train.csv",
        "target": "MedHouseVal",
        "type": "regression",
        "desc": "Predict California median house values from demographic and geographic features.",
    },
    "25_binary-prediction-rainfall": {
        "title": "Binary Rainfall Prediction",
        "url": "https://www.kaggle.com/competitions/playground-series-s5e3",
        "file": "train.csv",
        "target": "RainTomorrow",
        "type": "clf_imbalanced",
        "desc": "Predict whether it will rain tomorrow based on weather measurements.",
    },
}


# ── cell builders ─────────────────────────────────────────────────────────────
def md(source: str, cid: str) -> dict:
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": source}


def code(source: str, cid: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ── template functions ────────────────────────────────────────────────────────
def clf_notebook(p: dict, imbalanced: bool = False) -> dict:
    target = p["target"]
    title = p["title"]
    url = p["url"]
    fname = p["file"]
    desc = p["desc"]
    cw = ', class_weight="balanced"' if imbalanced else ""
    pr_auc = (
        "\n# Precision-Recall AUC (better for imbalanced data)\n"
        "from sklearn.metrics import average_precision_score\n"
        "pr_auc = average_precision_score(y_test, best_probs)\n"
        'print(f"PR-AUC: {pr_auc:.4f}")'
        if imbalanced
        else ""
    )

    cells = [
        md(
            f"# {title}\n\n"
            f"{desc}\n\n"
            f"**Dataset:** [{url}]({url})  \n"
            f"**Target:** `{target}`  \n"
            f"**Type:** {'Imbalanced ' if imbalanced else ''}Binary Classification\n\n"
            "> **TODO:** Download the dataset, place it in `../../data/raw/`, "
            "then update `DATA_PATH` and `TARGET` below.",
            "c00",
        ),
        code(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n"
            "from sklearn.metrics import (\n"
            "    classification_report, roc_auc_score,\n"
            "    roc_curve, ConfusionMatrixDisplay,\n"
            ")\n"
            "sns.set_theme(style='whitegrid')",
            "c01",
        ),
        md("## 1. Load Data", "c02"),
        code(
            f'# TODO: update path after downloading from {url}\n'
            f'DATA_PATH = "../../data/raw/{fname}"\n'
            f'TARGET = "{target}"  # TODO: verify column name\n\n'
            "df = pd.read_csv(DATA_PATH)\n"
            "print(f'Shape: {df.shape}')\n"
            "df.head()",
            "c03",
        ),
        md("## 2. Exploratory Data Analysis", "c04"),
        code(
            "print(df.info())\n"
            "print('\\nNull counts:')\n"
            "print(df.isnull().sum().sort_values(ascending=False).head(15))\n"
            "df.describe(include='all').T",
            "c05",
        ),
        code(
            "# Target distribution\n"
            "fig, ax = plt.subplots()\n"
            "df[TARGET].value_counts().plot(kind='bar', ax=ax)\n"
            "ax.set_title(f'Target distribution: {TARGET}')\n"
            "ax.set_xlabel(TARGET); ax.set_ylabel('Count')\n"
            "plt.xticks(rotation=0)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(df[TARGET].value_counts(normalize=True).round(3))",
            "c06",
        ),
        code(
            "# Correlation heatmap (numeric features)\n"
            "num_df = df.select_dtypes(include='number')\n"
            "if len(num_df.columns) > 1:\n"
            "    plt.figure(figsize=(10, 6))\n"
            "    sns.heatmap(num_df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)\n"
            "    plt.title('Correlation Matrix')\n"
            "    plt.tight_layout(); plt.show()",
            "c07",
        ),
        md("## 3. Feature Engineering", "c08"),
        code(
            "X = df.drop(columns=[TARGET])\n"
            "y = df[TARGET]\n\n"
            "# TODO: encode binary string targets if needed, e.g.:\n"
            "# y = y.map({'Yes': 1, 'No': 0})\n\n"
            "numeric_cols = X.select_dtypes(include=['number']).columns.tolist()\n"
            "categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()\n"
            "print('Numeric cols:', numeric_cols)\n"
            "print('Categorical cols:', categorical_cols)",
            "c09",
        ),
        code(
            "numeric_pipeline = Pipeline([\n"
            "    ('imputer', SimpleImputer(strategy='median')),\n"
            "    ('scaler', StandardScaler()),\n"
            "])\n"
            "categorical_pipeline = Pipeline([\n"
            "    ('imputer', SimpleImputer(strategy='most_frequent')),\n"
            "    ('onehot', OneHotEncoder(handle_unknown='ignore')),\n"
            "])\n"
            "preprocessor = ColumnTransformer([\n"
            "    ('num', numeric_pipeline, numeric_cols),\n"
            "    ('cat', categorical_pipeline, categorical_cols),\n"
            "])",
            "c10",
        ),
        md("## 4. Train / Test Split", "c11"),
        code(
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42, stratify=y\n"
            ")\n"
            "print(f'Train: {X_train.shape}, Test: {X_test.shape}')",
            "c12",
        ),
        md("## 5. Model Training", "c13"),
        code(
            "models = {\n"
            f'    "Logistic Regression": LogisticRegression(max_iter=1000{cw}),\n'
            f'    "Random Forest": RandomForestClassifier(n_estimators=100{cw}, random_state=42),\n'
            '    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),\n'
            "}\n\n"
            "results = {}\n"
            "for name, clf in models.items():\n"
            "    pipe = Pipeline([('preprocessor', preprocessor), ('clf', clf)])\n"
            "    pipe.fit(X_train, y_train)\n"
            "    preds = pipe.predict(X_test)\n"
            "    probs = pipe.predict_proba(X_test)[:, 1]\n"
            "    auc = roc_auc_score(y_test, probs)\n"
            "    results[name] = {'pipe': pipe, 'preds': preds, 'probs': probs, 'roc_auc': auc}\n"
            "    print(f'\\n=== {name} ===')\n"
            "    print(f'ROC-AUC: {auc:.4f}')\n"
            "    print(classification_report(y_test, preds))",
            "c14",
        ),
        md("## 6. Evaluation", "c15"),
        code(
            "best_name = max(results, key=lambda k: results[k]['roc_auc'])\n"
            "best = results[best_name]\n"
            "best_probs = best['probs']\n"
            "print(f'Best model: {best_name}  ROC-AUC: {best[\"roc_auc\"]:.4f}')\n"
            + pr_auc,
            "c16",
        ),
        code(
            "# Confusion Matrix\n"
            "ConfusionMatrixDisplay.from_predictions(y_test, best['preds'])\n"
            "plt.title(f'Confusion Matrix — {best_name}')\n"
            "plt.show()",
            "c17",
        ),
        code(
            "# ROC Curves\n"
            "fig, ax = plt.subplots()\n"
            "for name, res in results.items():\n"
            "    fpr, tpr, _ = roc_curve(y_test, res['probs'])\n"
            "    ax.plot(fpr, tpr, label=f\"{name} (AUC={res['roc_auc']:.3f})\")\n"
            "ax.plot([0, 1], [0, 1], 'k--', label='Random')\n"
            "ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')\n"
            "ax.legend(); ax.set_title('ROC Curves')\n"
            "plt.tight_layout(); plt.show()",
            "c18",
        ),
        code(
            "# Feature importances (Random Forest)\n"
            "rf_pipe = results['Random Forest']['pipe']\n"
            "rf_clf = rf_pipe.named_steps['clf']\n"
            "feat_names = (\n"
            "    rf_pipe.named_steps['preprocessor']\n"
            "    .get_feature_names_out()\n"
            ")\n"
            "importances = pd.Series(rf_clf.feature_importances_, index=feat_names)\n"
            "importances.nlargest(15).sort_values().plot(kind='barh', figsize=(8, 5))\n"
            "plt.title('Top 15 Feature Importances (Random Forest)')\n"
            "plt.tight_layout(); plt.show()",
            "c19",
        ),
        md(
            "## 7. Conclusion\n\n"
            "| Model | ROC-AUC |\n"
            "|---|---|\n"
            "| *(fill after running)* | |\n\n"
            "**Observations:**\n- \n\n"
            "**Next steps:**\n- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)\n"
            "- Try XGBoost / LightGBM\n"
            "- Threshold optimisation for Precision/Recall trade-off",
            "c20",
        ),
    ]
    return notebook(cells)


def regression_notebook(p: dict) -> dict:
    target = p["target"]
    title = p["title"]
    url = p["url"]
    fname = p["file"]
    desc = p["desc"]

    cells = [
        md(
            f"# {title}\n\n"
            f"{desc}\n\n"
            f"**Dataset:** [{url}]({url})  \n"
            f"**Target:** `{target}`  \n"
            "**Type:** Regression\n\n"
            "> **TODO:** Download the dataset, place it in `../../data/raw/`, "
            "then update `DATA_PATH` and `TARGET` below.",
            "c00",
        ),
        code(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.linear_model import LinearRegression, Ridge\n"
            "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n"
            "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
            "sns.set_theme(style='whitegrid')",
            "c01",
        ),
        md("## 1. Load Data", "c02"),
        code(
            f'# TODO: update path after downloading from {url}\n'
            f'DATA_PATH = "../../data/raw/{fname}"\n'
            f'TARGET = "{target}"  # TODO: verify column name\n\n'
            "df = pd.read_csv(DATA_PATH)\n"
            "print(f'Shape: {df.shape}')\n"
            "df.head()",
            "c03",
        ),
        md("## 2. Exploratory Data Analysis", "c04"),
        code(
            "print(df.info())\n"
            "print('\\nNull counts:')\n"
            "print(df.isnull().sum().sort_values(ascending=False).head(15))\n"
            "df.describe().T",
            "c05",
        ),
        code(
            "# Target distribution\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "df[TARGET].hist(bins=40, ax=axes[0])\n"
            "axes[0].set_title(f'Distribution: {TARGET}')\n"
            "np.log1p(df[TARGET]).hist(bins=40, ax=axes[1])\n"
            "axes[1].set_title(f'Log Distribution: {TARGET}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(df[TARGET].describe())",
            "c06",
        ),
        code(
            "# Correlation with target\n"
            "num_df = df.select_dtypes(include='number')\n"
            "corr = num_df.corr()[TARGET].drop(TARGET).sort_values()\n"
            "corr.plot(kind='barh', figsize=(8, max(4, len(corr) * 0.3)))\n"
            "plt.title(f'Feature Correlation with {TARGET}')\n"
            "plt.tight_layout(); plt.show()",
            "c07",
        ),
        md("## 3. Feature Engineering", "c08"),
        code(
            "X = df.drop(columns=[TARGET])\n"
            "y = df[TARGET]\n\n"
            "# Optional: log-transform skewed target\n"
            "# y = np.log1p(y)\n\n"
            "numeric_cols = X.select_dtypes(include=['number']).columns.tolist()\n"
            "categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()\n\n"
            "numeric_pipeline = Pipeline([\n"
            "    ('imputer', SimpleImputer(strategy='median')),\n"
            "    ('scaler', StandardScaler()),\n"
            "])\n"
            "categorical_pipeline = Pipeline([\n"
            "    ('imputer', SimpleImputer(strategy='most_frequent')),\n"
            "    ('onehot', OneHotEncoder(handle_unknown='ignore')),\n"
            "])\n"
            "preprocessor = ColumnTransformer([\n"
            "    ('num', numeric_pipeline, numeric_cols),\n"
            "    ('cat', categorical_pipeline, categorical_cols),\n"
            "])",
            "c09",
        ),
        md("## 4. Train / Test Split", "c10"),
        code(
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42\n"
            ")\n"
            "print(f'Train: {X_train.shape}, Test: {X_test.shape}')",
            "c11",
        ),
        md("## 5. Model Training", "c12"),
        code(
            "def eval_reg(name, pipe, X_tr, X_te, y_tr, y_te):\n"
            "    pipe.fit(X_tr, y_tr)\n"
            "    preds = pipe.predict(X_te)\n"
            "    mae = mean_absolute_error(y_te, preds)\n"
            "    rmse = np.sqrt(mean_squared_error(y_te, preds))\n"
            "    r2 = r2_score(y_te, preds)\n"
            "    print(f'{name}: MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}')\n"
            "    return pipe, preds, {'mae': mae, 'rmse': rmse, 'r2': r2}\n\n"
            "models = {\n"
            "    'Linear Regression': Pipeline([('pre', preprocessor), ('reg', LinearRegression())]),\n"
            "    'Ridge': Pipeline([('pre', preprocessor), ('reg', Ridge(alpha=1.0))]),\n"
            "    'Random Forest': Pipeline([('pre', preprocessor),\n"
            "                              ('reg', RandomForestRegressor(n_estimators=100, random_state=42))]),\n"
            "    'Gradient Boosting': Pipeline([('pre', preprocessor),\n"
            "                                  ('reg', GradientBoostingRegressor(n_estimators=100, random_state=42))]),\n"
            "}\n\n"
            "results = {}\n"
            "for name, pipe in models.items():\n"
            "    fitted, preds, metrics = eval_reg(name, pipe, X_train, X_test, y_train, y_test)\n"
            "    results[name] = {'pipe': fitted, 'preds': preds, 'metrics': metrics}",
            "c13",
        ),
        md("## 6. Evaluation", "c14"),
        code(
            "best_name = min(results, key=lambda k: results[k]['metrics']['rmse'])\n"
            "best_preds = results[best_name]['preds']\n"
            "print(f'Best model: {best_name}')\n\n"
            "# Actual vs Predicted\n"
            "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n"
            "axes[0].scatter(y_test, best_preds, alpha=0.4, s=15)\n"
            "lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]\n"
            "axes[0].plot(lims, lims, 'r--')\n"
            "axes[0].set_xlabel('Actual'); axes[0].set_ylabel('Predicted')\n"
            "axes[0].set_title(f'Actual vs Predicted — {best_name}')\n\n"
            "# Residuals\n"
            "residuals = y_test - best_preds\n"
            "axes[1].scatter(best_preds, residuals, alpha=0.4, s=15)\n"
            "axes[1].axhline(0, color='r', linestyle='--')\n"
            "axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Residual')\n"
            "axes[1].set_title('Residual Plot')\n"
            "plt.tight_layout(); plt.show()",
            "c15",
        ),
        code(
            "# Feature importances\n"
            "rf_pipe = results['Random Forest']['pipe']\n"
            "feat_names = rf_pipe.named_steps['pre'].get_feature_names_out()\n"
            "importances = pd.Series(\n"
            "    rf_pipe.named_steps['reg'].feature_importances_, index=feat_names\n"
            ")\n"
            "importances.nlargest(15).sort_values().plot(kind='barh', figsize=(8, 5))\n"
            "plt.title('Top 15 Feature Importances (Random Forest)')\n"
            "plt.tight_layout(); plt.show()",
            "c16",
        ),
        md(
            "## 7. Conclusion\n\n"
            "| Model | MAE | RMSE | R² |\n"
            "|---|---|---|---|\n"
            "| *(fill after running)* | | | |\n\n"
            "**Observations:**\n- \n\n"
            "**Next steps:**\n"
            "- Log-transform skewed target if not done\n"
            "- Hyperparameter tuning\n"
            "- Try XGBoost / LightGBM",
            "c17",
        ),
    ]
    return notebook(cells)


def timeseries_notebook(p: dict) -> dict:
    target = p["target"]
    title = p["title"]
    url = p["url"]
    fname = p["file"]
    desc = p["desc"]
    date_col = p.get("date_col", "date")

    cells = [
        md(
            f"# {title}\n\n"
            f"{desc}\n\n"
            f"**Dataset:** [{url}]({url})  \n"
            f"**Target:** `{target}`  \n"
            "**Type:** Time Series Forecasting\n\n"
            "> **TODO:** Download the dataset, place it in `../../data/raw/`, "
            "then update `DATA_PATH`, `DATE_COL`, and `TARGET` below.",
            "c00",
        ),
        code(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.linear_model import LinearRegression\n"
            "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n"
            "from sklearn.metrics import mean_absolute_error, mean_squared_error\n"
            "sns.set_theme(style='whitegrid')",
            "c01",
        ),
        md("## 1. Load & Parse Dates", "c02"),
        code(
            f'DATA_PATH = "../../data/raw/{fname}"\n'
            f'DATE_COL = "{date_col}"  # TODO: verify date column name\n'
            f'TARGET = "{target}"      # TODO: verify target column name\n\n'
            "df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])\n"
            "df = df.sort_values(DATE_COL).reset_index(drop=True)\n"
            "print(f'Shape: {df.shape}')\n"
            "print(f'Date range: {df[DATE_COL].min()} → {df[DATE_COL].max()}')\n"
            "df.head()",
            "c03",
        ),
        md("## 2. Time Series EDA", "c04"),
        code(
            "# Overall trend\n"
            "plt.figure(figsize=(14, 4))\n"
            "plt.plot(df[DATE_COL], df[TARGET], linewidth=0.8)\n"
            "plt.title(f'{TARGET} over time')\n"
            "plt.xlabel(DATE_COL); plt.ylabel(TARGET)\n"
            "plt.tight_layout(); plt.show()\n\n"
            "print(df[TARGET].describe())",
            "c05",
        ),
        code(
            "# Seasonal patterns\n"
            "df['year'] = df[DATE_COL].dt.year\n"
            "df['month'] = df[DATE_COL].dt.month\n"
            "df['dayofweek'] = df[DATE_COL].dt.dayofweek\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
            "df.groupby('month')[TARGET].mean().plot(ax=axes[0])\n"
            "axes[0].set_title('Average by Month')\n"
            "df.groupby('dayofweek')[TARGET].mean().plot(ax=axes[1])\n"
            "axes[1].set_title('Average by Day of Week')\n"
            "plt.tight_layout(); plt.show()",
            "c06",
        ),
        md("## 3. Feature Engineering", "c07"),
        code(
            "# Calendar features\n"
            "df['quarter'] = df[DATE_COL].dt.quarter\n"
            "df['weekofyear'] = df[DATE_COL].dt.isocalendar().week.astype(int)\n\n"
            "# Lag features — adjust window sizes to your data frequency\n"
            "for lag in [1, 7, 14, 28]:\n"
            "    df[f'lag_{lag}'] = df[TARGET].shift(lag)\n\n"
            "# Rolling statistics\n"
            "for window in [7, 14]:\n"
            "    df[f'rolling_mean_{window}'] = df[TARGET].shift(1).rolling(window).mean()\n"
            "    df[f'rolling_std_{window}'] = df[TARGET].shift(1).rolling(window).std()\n\n"
            "df = df.dropna().reset_index(drop=True)\n"
            "print(f'Shape after feature engineering: {df.shape}')",
            "c08",
        ),
        md("## 4. Time-Based Train / Test Split", "c09"),
        code(
            "# Use last 20% of time as test set (never shuffle time series!)\n"
            "split_idx = int(len(df) * 0.8)\n"
            "train_df = df.iloc[:split_idx]\n"
            "test_df = df.iloc[split_idx:]\n\n"
            "drop_cols = [TARGET, DATE_COL]\n"
            "feature_cols = [c for c in df.columns if c not in drop_cols]\n\n"
            "X_train, y_train = train_df[feature_cols], train_df[TARGET]\n"
            "X_test, y_test = test_df[feature_cols], test_df[TARGET]\n"
            "print(f'Train: {X_train.shape}, Test: {X_test.shape}')",
            "c10",
        ),
        md("## 5. Model Training", "c11"),
        code(
            "def mape(y_true, y_pred):\n"
            "    return np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-9, y_true))) * 100\n\n"
            "models = {\n"
            "    'Linear Regression': LinearRegression(),\n"
            "    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),\n"
            "    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),\n"
            "}\n\n"
            "results = {}\n"
            "for name, model in models.items():\n"
            "    model.fit(X_train, y_train)\n"
            "    preds = model.predict(X_test)\n"
            "    mae = mean_absolute_error(y_test, preds)\n"
            "    rmse = np.sqrt(mean_squared_error(y_test, preds))\n"
            "    mp = mape(y_test.values, preds)\n"
            "    results[name] = {'model': model, 'preds': preds}\n"
            "    print(f'{name}: MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mp:.2f}%')",
            "c12",
        ),
        md("## 6. Forecast Plot", "c13"),
        code(
            "best_name = min(\n"
            "    results,\n"
            "    key=lambda k: mean_squared_error(y_test, results[k]['preds'])\n"
            ")\n"
            "best_preds = results[best_name]['preds']\n\n"
            "plt.figure(figsize=(14, 5))\n"
            "plt.plot(test_df[DATE_COL].values, y_test.values, label='Actual', linewidth=1)\n"
            "plt.plot(test_df[DATE_COL].values, best_preds, label=f'Predicted ({best_name})',\n"
            "         linewidth=1, linestyle='--')\n"
            "plt.title(f'Forecast vs Actual — {best_name}')\n"
            "plt.xlabel('Date'); plt.ylabel(TARGET)\n"
            "plt.legend(); plt.tight_layout(); plt.show()",
            "c14",
        ),
        md(
            "## 7. Conclusion\n\n"
            "| Model | MAE | RMSE | MAPE |\n"
            "|---|---|---|---|\n"
            "| *(fill after running)* | | | |\n\n"
            "**Observations:**\n- \n\n"
            "**Next steps:**\n"
            "- Add more lag windows\n"
            "- Try cross-validation with TimeSeriesSplit\n"
            "- Explore SARIMA / Prophet for pure time-series approaches",
            "c15",
        ),
    ]
    return notebook(cells)


def clustering_notebook(p: dict) -> dict:
    title = p["title"]
    url = p["url"]
    fname = p["file"]
    desc = p["desc"]

    cells = [
        md(
            f"# {title}\n\n"
            f"{desc}\n\n"
            f"**Dataset:** [{url}]({url})  \n"
            "**Type:** Unsupervised Clustering\n\n"
            "> **TODO:** Download the dataset, place it in `../../data/raw/`, "
            "then update `DATA_PATH` below.",
            "c00",
        ),
        code(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.cluster import KMeans\n"
            "from sklearn.decomposition import PCA\n"
            "from sklearn.metrics import silhouette_score, davies_bouldin_score\n"
            "from sklearn.impute import SimpleImputer\n"
            "sns.set_theme(style='whitegrid')",
            "c01",
        ),
        md("## 1. Load Data", "c02"),
        code(
            f'DATA_PATH = "../../data/raw/{fname}"\n\n'
            "df = pd.read_csv(DATA_PATH)\n"
            "print(f'Shape: {df.shape}')\n"
            "df.head()",
            "c03",
        ),
        md("## 2. EDA", "c04"),
        code(
            "print(df.info())\n"
            "print('\\nNull counts:')\n"
            "print(df.isnull().sum())\n"
            "df.describe().T",
            "c05",
        ),
        code(
            "# Pairplot of numeric features (sample if large)\n"
            "num_df = df.select_dtypes(include='number')\n"
            "sample = num_df.sample(min(500, len(num_df)), random_state=42)\n"
            "sns.pairplot(sample, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})\n"
            "plt.suptitle('Feature Pairplot', y=1.01)\n"
            "plt.tight_layout(); plt.show()",
            "c06",
        ),
        md("## 3. Feature Selection & Scaling", "c07"),
        code(
            "# TODO: Select relevant features for clustering\n"
            "# Drop ID / date columns if present\n"
            "feature_cols = df.select_dtypes(include='number').columns.tolist()\n"
            "# feature_cols = ['col1', 'col2', ...]  # or specify manually\n\n"
            "X = df[feature_cols].copy()\n"
            "X = SimpleImputer(strategy='median').fit_transform(X)\n"
            "X_scaled = StandardScaler().fit_transform(X)\n"
            "print(f'Feature matrix shape: {X_scaled.shape}')",
            "c08",
        ),
        md("## 4. Determine Optimal K", "c09"),
        code(
            "inertias, silhouettes = [], []\n"
            "K_range = range(2, 11)\n\n"
            "for k in K_range:\n"
            "    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n"
            "    labels = km.fit_predict(X_scaled)\n"
            "    inertias.append(km.inertia_)\n"
            "    silhouettes.append(silhouette_score(X_scaled, labels))\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "axes[0].plot(list(K_range), inertias, 'bo-')\n"
            "axes[0].set_title('Elbow Method'); axes[0].set_xlabel('k')\n"
            "axes[1].plot(list(K_range), silhouettes, 'ro-')\n"
            "axes[1].set_title('Silhouette Score'); axes[1].set_xlabel('k')\n"
            "plt.tight_layout(); plt.show()\n\n"
            "best_k = list(K_range)[silhouettes.index(max(silhouettes))]\n"
            "print(f'Best k by silhouette: {best_k}')",
            "c10",
        ),
        md("## 5. Final Clustering", "c11"),
        code(
            "K = best_k  # TODO: override if domain knowledge suggests otherwise\n\n"
            "km_final = KMeans(n_clusters=K, random_state=42, n_init=10)\n"
            "df['cluster'] = km_final.fit_predict(X_scaled)\n\n"
            "sil = silhouette_score(X_scaled, df['cluster'])\n"
            "db = davies_bouldin_score(X_scaled, df['cluster'])\n"
            "print(f'Silhouette Score: {sil:.4f}')\n"
            "print(f'Davies-Bouldin Score: {db:.4f}  (lower is better)')\n"
            "print(df['cluster'].value_counts())",
            "c12",
        ),
        md("## 6. PCA Visualisation", "c13"),
        code(
            "pca = PCA(n_components=2)\n"
            "X_pca = pca.fit_transform(X_scaled)\n"
            "explained = pca.explained_variance_ratio_.sum()\n\n"
            "plt.figure(figsize=(8, 6))\n"
            "scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],\n"
            "                     c=df['cluster'], cmap='tab10', alpha=0.6, s=15)\n"
            "plt.colorbar(scatter, label='Cluster')\n"
            "plt.title(f'PCA Projection — {K} Clusters (var explained: {explained:.1%})')\n"
            "plt.xlabel('PC1'); plt.ylabel('PC2')\n"
            "plt.tight_layout(); plt.show()",
            "c14",
        ),
        md("## 7. Segment Profiles", "c15"),
        code(
            "cluster_profile = df.groupby('cluster')[feature_cols].mean().T\n"
            "cluster_profile.columns = [f'Cluster {c}' for c in cluster_profile.columns]\n"
            "print(cluster_profile.round(2))\n\n"
            "# Heatmap\n"
            "plt.figure(figsize=(10, max(4, len(feature_cols) * 0.4)))\n"
            "sns.heatmap(cluster_profile, cmap='RdYlGn', annot=True, fmt='.2f', linewidths=0.5)\n"
            "plt.title('Cluster Profiles (mean feature values)')\n"
            "plt.tight_layout(); plt.show()",
            "c16",
        ),
        md(
            "## 8. Conclusion\n\n"
            "| Cluster | Size | Interpretation |\n"
            "|---|---|---|\n"
            "| *(fill after running)* | | |\n\n"
            "**Observations:**\n- \n\n"
            "**Next steps:**\n"
            "- Try DBSCAN for density-based clustering\n"
            "- Add categorical features via Gower distance\n"
            "- Use clusters for downstream classification/regression tasks",
            "c17",
        ),
    ]
    return notebook(cells)


def recommender_notebook(p: dict) -> dict:
    title = p["title"]
    url = p["url"]
    fname = p["file"]
    desc = p["desc"]
    user_col = p.get("user_col", "userId")
    item_col = p.get("item_col", "movieId")
    target = p["target"]

    cells = [
        md(
            f"# {title}\n\n"
            f"{desc}\n\n"
            f"**Dataset:** [{url}]({url})  \n"
            f"**Columns:** `{user_col}`, `{item_col}`, `{target}`  \n"
            "**Type:** Collaborative Filtering\n\n"
            "> **TODO:** Download the dataset, place it in `../../data/raw/`, "
            "then update `DATA_PATH` below.",
            "c00",
        ),
        code(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.metrics.pairwise import cosine_similarity\n"
            "from sklearn.model_selection import train_test_split\n"
            "sns.set_theme(style='whitegrid')",
            "c01",
        ),
        md("## 1. Load Data", "c02"),
        code(
            f'DATA_PATH = "../../data/raw/{fname}"\n'
            f'USER_COL = "{user_col}"\n'
            f'ITEM_COL = "{item_col}"\n'
            f'RATING_COL = "{target}"\n\n'
            "df = pd.read_csv(DATA_PATH)\n"
            "# Use a sample for faster experimentation\n"
            "df = df.sample(min(100_000, len(df)), random_state=42).reset_index(drop=True)\n"
            "print(f'Shape: {df.shape}')\n"
            "print(f'Users: {df[USER_COL].nunique()}, Items: {df[ITEM_COL].nunique()}')\n"
            "df.head()",
            "c03",
        ),
        md("## 2. EDA", "c04"),
        code(
            "# Rating distribution\n"
            "df[RATING_COL].value_counts().sort_index().plot(kind='bar', figsize=(8, 4))\n"
            "plt.title('Rating Distribution'); plt.xlabel('Rating')\n"
            "plt.tight_layout(); plt.show()\n\n"
            "# Ratings per user / item\n"
            "ratings_per_user = df.groupby(USER_COL).size()\n"
            "ratings_per_item = df.groupby(ITEM_COL).size()\n"
            "print(f'Ratings per user: mean={ratings_per_user.mean():.1f}, median={ratings_per_user.median():.1f}')\n"
            "print(f'Ratings per item: mean={ratings_per_item.mean():.1f}, median={ratings_per_item.median():.1f}')",
            "c05",
        ),
        md("## 3. Train / Test Split", "c06"),
        code(
            "train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)\n"
            "print(f'Train: {len(train_df)}, Test: {len(test_df)}')",
            "c07",
        ),
        md("## 4. Popularity Baseline", "c08"),
        code(
            "# Recommend top-K most popular items to everyone\n"
            "K = 10\n"
            "popular_items = (\n"
            "    train_df.groupby(ITEM_COL)[RATING_COL].mean()\n"
            "    .sort_values(ascending=False)\n"
            "    .head(K)\n"
            "    .index.tolist()\n"
            ")\n"
            "print(f'Top-{K} popular items: {popular_items[:5]}...')",
            "c09",
        ),
        md("## 5. User-Item Matrix", "c10"),
        code(
            "# Build matrix (may be large — filter active users/items for experiments)\n"
            "min_ratings_user = 5\n"
            "min_ratings_item = 5\n"
            "active_users = ratings_per_user[ratings_per_user >= min_ratings_user].index\n"
            "popular_items_all = ratings_per_item[ratings_per_item >= min_ratings_item].index\n"
            "filtered = train_df[\n"
            "    train_df[USER_COL].isin(active_users) & train_df[ITEM_COL].isin(popular_items_all)\n"
            "]\n\n"
            "user_item_matrix = filtered.pivot_table(\n"
            "    index=USER_COL, columns=ITEM_COL, values=RATING_COL, fill_value=0\n"
            ")\n"
            "sparsity = 1 - (filtered.shape[0] / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))\n"
            "print(f'Matrix shape: {user_item_matrix.shape}')\n"
            "print(f'Sparsity: {sparsity:.1%}')",
            "c11",
        ),
        md("## 6. User-Based Collaborative Filtering", "c12"),
        code(
            "user_sim = cosine_similarity(user_item_matrix)\n"
            "user_sim_df = pd.DataFrame(\n"
            "    user_sim, index=user_item_matrix.index, columns=user_item_matrix.index\n"
            ")\n\n"
            "def recommend_user_based(user_id, n=10):\n"
            "    if user_id not in user_sim_df.index:\n"
            "        return popular_items[:n]  # cold-start fallback\n"
            "    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:21].index\n"
            "    seen = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)\n"
            "    scores = user_item_matrix.loc[similar_users].mean().drop(index=list(seen), errors='ignore')\n"
            "    return scores.nlargest(n).index.tolist()\n\n"
            "sample_user = user_item_matrix.index[0]\n"
            "recs = recommend_user_based(sample_user)\n"
            "print(f'Recommendations for user {sample_user}: {recs}')",
            "c13",
        ),
        md("## 7. Item-Based Collaborative Filtering", "c14"),
        code(
            "item_sim = cosine_similarity(user_item_matrix.T)\n"
            "item_sim_df = pd.DataFrame(\n"
            "    item_sim, index=user_item_matrix.columns, columns=user_item_matrix.columns\n"
            ")\n\n"
            "def recommend_item_based(user_id, n=10):\n"
            "    if user_id not in user_item_matrix.index:\n"
            "        return popular_items[:n]\n"
            "    rated = user_item_matrix.loc[user_id]\n"
            "    rated_items = rated[rated > 0].index\n"
            "    scores = item_sim_df[rated_items].mean(axis=1)\n"
            "    scores = scores.drop(index=rated_items, errors='ignore')\n"
            "    return scores.nlargest(n).index.tolist()\n\n"
            "recs_item = recommend_item_based(sample_user)\n"
            "print(f'Item-based recs for user {sample_user}: {recs_item}')",
            "c15",
        ),
        md("## 8. Offline Evaluation (Precision@K)", "c16"),
        code(
            "def precision_at_k(recommend_fn, test_df, k=10, n_users=200):\n"
            "    test_users = test_df[USER_COL].unique()[:n_users]\n"
            "    precisions = []\n"
            "    for uid in test_users:\n"
            "        relevant = set(test_df[test_df[USER_COL] == uid][ITEM_COL])\n"
            "        if not relevant:\n"
            "            continue\n"
            "        recs = set(recommend_fn(uid, n=k))\n"
            "        precisions.append(len(recs & relevant) / k)\n"
            "    return np.mean(precisions) if precisions else 0.0\n\n"
            "p_user = precision_at_k(recommend_user_based, test_df)\n"
            "p_item = precision_at_k(recommend_item_based, test_df)\n"
            "print(f'User-based Precision@10: {p_user:.4f}')\n"
            "print(f'Item-based Precision@10: {p_item:.4f}')",
            "c17",
        ),
        md(
            "## 9. Conclusion\n\n"
            "| Method | Precision@10 |\n"
            "|---|---|\n"
            "| Popularity Baseline | — |\n"
            "| User-Based CF | *(fill)* |\n"
            "| Item-Based CF | *(fill)* |\n\n"
            "**Observations:**\n- \n\n"
            "**Next steps:**\n"
            "- Try matrix factorization (SVD via `surprise` library)\n"
            "- Add content-based features (genres, tags)\n"
            "- Address cold-start with hybrid approach",
            "c18",
        ),
    ]
    return notebook(cells)


def text_clf_notebook(p: dict) -> dict:
    target = p["target"]
    title = p["title"]
    url = p["url"]
    fname = p["file"]
    desc = p["desc"]
    text_col = p.get("text_col", "text")

    cells = [
        md(
            f"# {title}\n\n"
            f"{desc}\n\n"
            f"**Dataset:** [{url}]({url})  \n"
            f"**Text column:** `{text_col}`  **Target:** `{target}`  \n"
            "**Type:** Binary Text Classification\n\n"
            "> **TODO:** Download the dataset, place it in `../../data/raw/`, "
            "then update `DATA_PATH`, `TEXT_COL`, and `TARGET` below.",
            "c00",
        ),
        code(
            "import re\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.feature_extraction.text import TfidfVectorizer\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.naive_bayes import MultinomialNB\n"
            "from sklearn.svm import LinearSVC\n"
            "from sklearn.metrics import (\n"
            "    classification_report, ConfusionMatrixDisplay,\n"
            "    roc_auc_score, roc_curve,\n"
            ")\n"
            "sns.set_theme(style='whitegrid')",
            "c01",
        ),
        md("## 1. Load Data", "c02"),
        code(
            f'DATA_PATH = "../../data/raw/{fname}"\n'
            f'TEXT_COL = "{text_col}"  # TODO: verify column name\n'
            f'TARGET = "{target}"       # TODO: verify column name\n\n'
            "df = pd.read_csv(DATA_PATH, encoding='latin-1')\n"
            "df = df[[TEXT_COL, TARGET]].dropna()\n"
            "print(f'Shape: {df.shape}')\n"
            "print(df[TARGET].value_counts())\n"
            "df.head()",
            "c03",
        ),
        md("## 2. EDA", "c04"),
        code(
            "# Class distribution\n"
            "df[TARGET].value_counts().plot(kind='bar')\n"
            "plt.title(f'Class Distribution: {TARGET}')\n"
            "plt.xticks(rotation=0); plt.tight_layout(); plt.show()\n\n"
            "# Text length\n"
            "df['text_len'] = df[TEXT_COL].str.len()\n"
            "df.groupby(TARGET)['text_len'].hist(alpha=0.6, bins=40)\n"
            "plt.title('Text Length by Class')\n"
            "plt.xlabel('Characters'); plt.tight_layout(); plt.show()\n"
            "print(df.groupby(TARGET)['text_len'].describe())",
            "c05",
        ),
        md("## 3. Text Preprocessing", "c06"),
        code(
            "def preprocess(text: str) -> str:\n"
            "    text = text.lower()\n"
            "    text = re.sub(r'<[^>]+>', ' ', text)  # strip HTML\n"
            "    text = re.sub(r'[^a-z0-9\\s]', ' ', text)\n"
            "    text = re.sub(r'\\s+', ' ', text).strip()\n"
            "    return text\n\n"
            "df['clean_text'] = df[TEXT_COL].astype(str).apply(preprocess)\n"
            "df[['clean_text']].head(3)",
            "c07",
        ),
        md("## 4. Train / Test Split", "c08"),
        code(
            "X = df['clean_text']\n"
            "y = df[TARGET]\n\n"
            "# TODO: encode labels if they are strings, e.g.:\n"
            "# y = y.map({'positive': 1, 'negative': 0})\n\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42, stratify=y\n"
            ")\n"
            "print(f'Train: {len(X_train)}, Test: {len(X_test)}')",
            "c09",
        ),
        md("## 5. Model Training", "c10"),
        code(
            "tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)\n\n"
            "models = {\n"
            "    'Naive Bayes': Pipeline([('tfidf', tfidf), ('clf', MultinomialNB())]),\n"
            "    'Logistic Regression': Pipeline([('tfidf', tfidf),\n"
            "                                    ('clf', LogisticRegression(max_iter=1000))]),\n"
            "    'LinearSVC': Pipeline([('tfidf', tfidf), ('clf', LinearSVC(max_iter=2000))]),\n"
            "}\n\n"
            "results = {}\n"
            "for name, pipe in models.items():\n"
            "    pipe.fit(X_train, y_train)\n"
            "    preds = pipe.predict(X_test)\n"
            "    results[name] = {'pipe': pipe, 'preds': preds}\n"
            "    print(f'\\n=== {name} ===')\n"
            "    print(classification_report(y_test, preds))",
            "c11",
        ),
        md("## 6. Evaluation", "c12"),
        code(
            "# Confusion matrices\n"
            "fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))\n"
            "for ax, (name, res) in zip(axes, results.items()):\n"
            "    ConfusionMatrixDisplay.from_predictions(y_test, res['preds'], ax=ax)\n"
            "    ax.set_title(name)\n"
            "plt.tight_layout(); plt.show()",
            "c13",
        ),
        code(
            "# Top TF-IDF terms per class (Logistic Regression)\n"
            "lr_pipe = results['Logistic Regression']['pipe']\n"
            "vocab = lr_pipe.named_steps['tfidf'].get_feature_names_out()\n"
            "coef = lr_pipe.named_steps['clf'].coef_[0]\n\n"
            "top_pos = pd.Series(coef, index=vocab).nlargest(15)\n"
            "top_neg = pd.Series(coef, index=vocab).nsmallest(15)\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
            "top_pos.sort_values().plot(kind='barh', ax=axes[0])\n"
            "axes[0].set_title('Top Positive Terms')\n"
            "top_neg.sort_values().plot(kind='barh', ax=axes[1])\n"
            "axes[1].set_title('Top Negative Terms')\n"
            "plt.tight_layout(); plt.show()",
            "c14",
        ),
        md(
            "## 7. Conclusion\n\n"
            "| Model | Accuracy | F1 |\n"
            "|---|---|---|\n"
            "| *(fill after running)* | | |\n\n"
            "**Observations:**\n- \n\n"
            "**Next steps:**\n"
            "- Tune `max_features` and `ngram_range` in TF-IDF\n"
            "- Try character-level n-grams for noisy text\n"
            "- Explore pre-trained embeddings (GloVe, fastText)",
            "c15",
        ),
    ]
    return notebook(cells)


# ── dispatch ──────────────────────────────────────────────────────────────────
def build_notebook(key: str, p: dict) -> dict:
    t = p["type"]
    if t == "clf_imbalanced":
        return clf_notebook(p, imbalanced=True)
    if t == "clf_standard":
        return clf_notebook(p, imbalanced=False)
    if t == "regression":
        return regression_notebook(p)
    if t == "timeseries":
        return timeseries_notebook(p)
    if t == "clustering":
        return clustering_notebook(p)
    if t == "recommender":
        return recommender_notebook(p)
    if t == "text_clf":
        return text_clf_notebook(p)
    raise ValueError(f"Unknown type: {t}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for folder, meta in PROJECTS.items():
        nb = build_notebook(folder, meta)
        out = BASE / folder / "starter_notebook.ipynb"
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  OK  {folder}/starter_notebook.ipynb")
    print(f"\nDone — {len(PROJECTS)} notebooks generated.")
