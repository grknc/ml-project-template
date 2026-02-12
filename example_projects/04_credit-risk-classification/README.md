## Project Overview
Credit risk classification helps lenders evaluate whether a loan applicant is likely to repay or default. Better risk estimation supports safer lending, reduced losses, and more consistent credit policies. The goal is to build a model that supports responsible and data-driven approval decisions.

## Problem Definition
The project predicts if an applicant belongs to a low-risk or high-risk credit group.
This is a supervised **classification** problem, often with class imbalance.

## Datasets
- **Dataset name:** Credit Risk Dataset
- **Source:** https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data
- **Description:** The dataset includes applicant financial and personal attributes, loan-related variables, and a target label indicating credit risk level.

## Recommended Models / Methods
- **Baseline:** Majority class model or Logistic Regression.
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine

## Evaluation Metrics
- **Recall:** Critical for identifying risky applicants.
- **Precision:** Reduces false positive risk flags.
- **F1-score:** Balances precision and recall.
- **ROC-AUC:** Evaluates ranking quality across thresholds.

## Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib / seaborn

## Expected Deliverables
- EDA notebook
- Feature engineering steps
- At least two model experiments
- Model evaluation and comparison
- Short business or real-world interpretation
