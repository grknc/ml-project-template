## 1) Project Overview
Credit risk classification helps financial institutions decide whether a loan applicant is likely to repay or default. Better risk assessment supports safer lending decisions.

## 2) Problem Definition
This is an imbalanced classification problem. The goal is to predict whether an applicant is low risk or high risk (default risk).

This problem matters because wrong lending decisions can create large financial losses. In this context, false negatives are costly: predicting a risky applicant as safe can lead to unpaid loans.

## 3) Suggested Datasets
You can use public credit risk datasets from platforms like Kaggle.

Typical datasets include:
- Applicant income and employment details
- Credit history and payment behavior
- Loan amount and term information
- Target label for default or non-default

## 4) Recommended Models
Suitable models for credit risk classification include:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting models (like XGBoost or LightGBM)
- Support Vector Machine

Because the data is imbalanced, consider class weighting or resampling techniques.

## 5) Evaluation Metrics
For imbalanced credit risk tasks, focus on:
- Recall: important for finding as many risky applicants as possible
- Precision: helps avoid too many false alerts
- F1-score: balances precision and recall

These metrics are better than accuracy alone in imbalanced settings.

## 6) Tools & Libraries
Recommended tools and libraries:
- Python
- pandas, numpy
- scikit-learn
- matplotlib / seaborn

## 7) Expected Deliverables
Your project outputs should include:
- EDA notebook
- Feature engineering
- At least 2 model experiments
- Model evaluation and comparison
- Business insights / conclusions
