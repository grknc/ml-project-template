# 08 CLTV Prediction

## 1) Project Overview
This project estimates Customer Lifetime Value (CLTV), which is the expected future value of a customer. The goal is to support smarter marketing, retention, and budget allocation decisions. By predicting CLTV, teams can focus on high-value segments and improve campaign efficiency. The project can combine practical regression modeling with customer analytics thinking.

## 2) Problem Definition
- **What is predicted or analyzed:** Future customer value over a chosen time horizon.
- **Target:** Numeric CLTV value (regression) or value segments based on expected contribution.
- **Typical stakeholders / business value:** Marketing, CRM, and finance teams use CLTV to prioritize campaigns and manage acquisition/retention spend.
- **Important note:** Insights from CLTV can directly influence budget planning and channel strategy.

## 3) Suggested Datasets
- **Online Retail II (UCI):** Transaction-level data often used for RFM and CLTV work.
- **E-Commerce Behavior Data (Kaggle variants):** Useful for building customer-level purchase features.
- **Customer Personality Analysis (Kaggle):** Includes campaign response and customer profile variables.

## 4) Recommended Models / Methods
- **Baseline approach:** Simple average historical spend per customer.
- Feature-based regression (Linear Regression, Random Forest Regressor, Gradient Boosting).
- XGBoost or LightGBM for non-linear customer value patterns.
- Optional concept method: probabilistic CLV with **BG/NBD + Gamma-Gamma**.
- Use customer-level features such as recency, frequency, monetary value, tenure, and engagement.

## 5) Evaluation Metrics
- **MAE / RMSE:** Core regression metrics for absolute and large-error sensitivity.
- Compare model quality across customer segments (for example, low/mid/high value groups).
- Segment-level validation helps check if predictions are useful for business actions.
- Also evaluate whether model-based targeting improves expected marketing ROI.

## 6) Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib / seaborn
- xgboost / lightgbm (optional)

## 7) Expected Deliverables
- EDA notebook on purchase behavior, RFM patterns, and segment differences.
- Feature engineering at customer level from transaction history.
- At least 2 experiments (for example, linear baseline vs boosted model).
- Model evaluation and comparison with segment-level analysis.
- Business insights / conclusions for marketing and budget decisions.
- Clear README + reproducible steps.
