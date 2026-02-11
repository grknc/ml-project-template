# 24 California House Price Regression

## 1) Project Overview
This project estimates house prices in California from tabular property and location features. It is a standard regression problem and a good case for comparing simple and regularized linear models. The objective is to build a reliable model with clear interpretation.

## 2) Problem Definition
- **What is predicted or analyzed:** House sale or listing price.
- **Target:** Continuous numeric price value.
- **Problem type:** Regression.
- **Key note:** Start with a baseline model, then test regularization approaches like Ridge and Lasso.

## 3) Suggested Datasets
- **Name:** Regression - California House Price (Tabular)
- **Dataset Source:** <https://www.kaggle.com/competitions/regression-tabular-california-housing/data?select=train.csv>
- Common features include location-related attributes, room counts, and population-based indicators.

## 4) Recommended Models / Methods
- **Baseline:** Linear Regression.
- Ridge Regression and Lasso Regression.
- Gradient Boosting Regressor.
- Compare models with and without feature scaling where needed.

## 5) Evaluation Metrics
- RMSE (Root Mean Squared Error).
- MAE (Mean Absolute Error).
- R2 (Coefficient of Determination).

## 6) Tools & Libraries
- Python.
- pandas, numpy.
- scikit-learn.
- matplotlib / seaborn.

## 7) Expected Deliverables
- Data quality and feature summary.
- Baseline and regularized model comparison.
- Error analysis and model interpretation notes.
- Final recommendation for the best model and trade-offs.

## 8) Common Pitfalls & Tips
- Begin with a simple baseline before tuning advanced models.
- Scale features when using regularized linear models.
- Check multicollinearity and extreme outliers.
- Do not rely on one metric only; compare RMSE, MAE, and R2 together.
- Keep validation strategy consistent for fair model comparison.
