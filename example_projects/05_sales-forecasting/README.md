# 05 Sales Forecasting

## 1) Project Overview
This project predicts future sales for a store, product group, or business unit over time. The goal is to help teams plan inventory, staffing, and promotions with fewer surprises. Sales forecasting is common in retail, e-commerce, and supply chain planning. A strong forecast can reduce stockouts, lower waste, and improve cash flow.

## 2) Problem Definition
- **What is predicted or analyzed:** Future sales values by day, week, or month.
- **Target:** A numeric sales amount for each future time period.
- **Typical stakeholders / business value:** Sales managers, supply chain teams, and finance teams use forecasts for planning and budget control.

## 3) Suggested Datasets
- **Store Item Demand Forecasting Challenge (Kaggle):** Daily sales by store and item, good for time series practice.
- **Corporación Favorita Grocery Sales Forecasting (Kaggle):** Real retail-style data with promotions and seasonality.
- **Rossmann Store Sales (Kaggle):** Store-level daily sales with marketing and holiday effects.

## 4) Recommended Models / Methods
- **Baseline approach:** Naive forecast (for example, last period's value) or simple moving average.
- Linear Regression with time-based features.
- Random Forest Regressor or Gradient Boosting Regressor.
- XGBoost or LightGBM for stronger tabular forecasting performance.
- Create **lag features** (past sales values) and **rolling statistics** (rolling mean, rolling std).
- Use a **time-based split** for train/validation (never random split for temporal data).

## 5) Evaluation Metrics
- **MAE:** Easy to understand average absolute error in sales units.
- **RMSE:** Penalizes large errors more strongly than MAE.
- **MAPE/SMAPE (optional):** Useful when relative error is important across products.
- Validate on future periods only to simulate real forecasting.
- Watch for **data leakage**: do not use future information when creating features.

## 6) Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib / seaborn
- xgboost / lightgbm (optional)

## 7) Expected Deliverables
- EDA notebook focused on trend, seasonality, and outliers.
- Feature engineering with calendar, lag, and rolling features.
- At least 2 experiments (for example, baseline vs tree-based model).
- Model evaluation and comparison on a time-based validation set.
- Business insights / conclusions (inventory, campaign timing, planning).
- Clear README + reproducible steps.
