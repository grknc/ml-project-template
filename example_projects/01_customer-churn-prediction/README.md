## Project Overview
Bank customer churn prediction helps banks identify clients who may close their accounts or stop using services. Early identification gives teams time to improve support, offer retention campaigns, and reduce revenue loss. The main goal is to support better customer retention decisions with data.

## Problem Definition
The project predicts whether a bank customer will churn or stay.
This is a supervised **classification** problem with a binary target.

## Datasets
- **Dataset name:** Bank Customer Churn Dataset
- **Source:** https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data
- **Description:** The dataset contains customer profile, account, and service usage features, along with a churn label that shows if the customer left.

## Recommended Models / Methods
- **Baseline:** Majority class prediction or a simple Logistic Regression model.
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting (e.g., XGBoost or LightGBM)
- Support Vector Machine

## Evaluation Metrics
- **Recall:** Important to catch as many real churners as possible.
- **Precision:** Helps reduce unnecessary retention actions.
- **F1-score:** Balances recall and precision.
- **ROC-AUC:** Measures how well the model separates churn vs non-churn.

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
