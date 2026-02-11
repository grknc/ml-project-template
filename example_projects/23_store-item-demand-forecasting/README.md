# 23 Store Item Demand Forecasting

## 1) Project Overview
This project predicts future item demand for stores using historical sales records. It is usually a time-series forecasting task, but can also be modeled as regression with engineered time features. The goal is to improve inventory planning and reduce stock issues.

## 2) Problem Definition
- **What is predicted or analyzed:** Future demand per store-item combination.
- **Target:** Numeric demand quantity.
- **Problem type:** Time-series forecasting / regression with lag features.
- **Key note:** Do not use random train/test split; always use time-based validation.

## 3) Suggested Datasets
- **Name:** Store Item Demand Forecasting Challenge (Kernels Only)
- **Dataset Source:** <https://www.kaggle.com/competitions/demand-forecasting-kernels-only>
- Typical fields include date, store id, item id, and sales.

## 4) Recommended Models / Methods
- **Baseline:** Naive last-value method or moving-average concept.
- Linear Regression with lag and calendar features.
- Random Forest.
- Gradient Boosting.
- Feature engineering focus: lag values, rolling means, and seasonality indicators.

## 5) Evaluation Metrics
- MAE (Mean Absolute Error).
- RMSE (Root Mean Squared Error).
- Optional: SMAPE for relative error comparison across scales.

## 6) Tools & Libraries
- Python.
- pandas, numpy.
- scikit-learn.
- matplotlib / seaborn.

## 7) Expected Deliverables
- Time-aware validation strategy description.
- Feature engineering notes for lag, rolling, and seasonal signals.
- Baseline and model benchmark comparison.
- Final forecasting report with practical inventory insights.

## 8) Common Pitfalls & Tips
- Never apply random split to time-ordered data.
- Create lag features carefully to avoid future leakage.
- Use a strong baseline before complex models.
- Check performance by different stores and item groups.
- Validate on multiple time windows, not only one split.
