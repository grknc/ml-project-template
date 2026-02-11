# 07 Fraud Detection

## 1) Project Overview
This project identifies suspicious transactions that may be fraudulent. Fraud detection helps organizations reduce financial loss and protect customers. In real settings, fraud cases are rare, so the model must work well on highly imbalanced data. The solution should support fast investigation and risk-based decision-making.

## 2) Problem Definition
- **What is predicted or analyzed:** Whether a transaction is legitimate or fraudulent.
- **Target:** Binary label (fraud: yes/no).
- **Typical stakeholders / business value:** Risk teams, fraud analysts, and operations teams use predictions to prioritize investigations and block high-risk activity.
- **Important note:** False negatives are costly because missed fraud leads to direct loss.

## 3) Suggested Datasets
- **Credit Card Fraud Detection (Kaggle):** Classic highly imbalanced transaction dataset.
- **IEEE-CIS Fraud Detection (Kaggle):** Large, realistic dataset with transaction and identity features.
- **PaySim synthetic mobile money dataset (Kaggle):** Simulated transaction flows useful for behavior-based modeling.

## 4) Recommended Models / Methods
- **Baseline approach:** Logistic Regression with class weights.
- Random Forest / Balanced Random Forest.
- Gradient Boosting, XGBoost, or LightGBM with imbalance-aware settings.
- Isolation-based methods can be tested as complementary anomaly signals.
- Apply threshold tuning to manage precision/recall trade-offs.

## 5) Evaluation Metrics
- **Precision:** Shows how many flagged transactions are truly fraud.
- **Recall:** Shows how much fraud is captured; often the top priority.
- **F1-score:** Useful summary when both precision and recall matter.
- **PR-AUC (optional):** Better than ROC-AUC in highly imbalanced settings.
- Validate with realistic class ratios and compare different decision thresholds.

## 6) Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib / seaborn
- xgboost / lightgbm (optional)

## 7) Expected Deliverables
- EDA notebook on class imbalance and transaction behavior.
- Feature engineering for frequency, amount patterns, and time signals.
- At least 2 experiments (for example, weighted baseline vs boosting).
- Model evaluation and comparison with threshold analysis.
- Business insights / conclusions on alert strategy and analyst workload.
- Clear README + reproducible steps.
