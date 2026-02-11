# 15 Loan Default Prediction

## 1) Project Overview
This project predicts whether a borrower may default on a loan. It is a financial risk classification problem. The main objective is to support safer lending decisions while reducing unnecessary rejection.

## 2) Problem Definition
- **What is predicted or analyzed:** Probability of loan default.
- **Target:** Binary class (default / no default).
- **Typical stakeholders / business value:** Risk analysts and lending teams use this to improve portfolio quality.
- **Common challenges:** Imbalanced data, missing financial values, and fairness concerns.

## 3) Suggested Datasets
- **Lending Club Loan Data (public subsets):** Loan and borrower-level variables.
- **German Credit Data (UCI):** Classic credit risk classification dataset.
- **Home Credit Default Risk (Kaggle):** Larger real-world-style credit features.

## 4) Recommended Models / Methods
- **Baseline approach:** Logistic Regression.
- Random Forest.
- Gradient Boosting.
- Use class weighting or resampling strategies to handle imbalanced data.

## 5) Evaluation Metrics
- Recall.
- F1-score.
- ROC-AUC.
- Include threshold discussion for risk-sensitive decisions.

## 6) Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- imbalanced-learn (optional)
- matplotlib / seaborn

## 7) Expected Deliverables
- Data quality and class imbalance analysis.
- Model comparison with imbalanced-data handling.
- Threshold and risk trade-off discussion.
- Key risk drivers and practical recommendations.
- Final report and reproducible README steps.
