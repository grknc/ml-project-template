## 1) Project Overview
Customer churn prediction helps a business identify customers who are likely to leave soon. If a company can find these customers early, it can take action to keep them with better offers or support.

## 2) Problem Definition
This is a classification problem. The goal is to predict whether a customer will churn (leave) or stay.

This problem matters because losing customers is expensive. Retaining existing customers is often cheaper than acquiring new ones. Better churn prediction can improve revenue and customer satisfaction.

## 3) Suggested Datasets
You can use public telecom or subscription datasets from platforms like Kaggle.

Good examples include datasets with:
- Customer demographics
- Service usage patterns
- Contract type and payment history
- Churn label (Yes/No)

## 4) Recommended Models
Suitable models for churn classification include:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting models (like XGBoost or LightGBM)
- Support Vector Machine

Because churn data is often imbalanced, also consider:
- Class weights
- Resampling methods (oversampling or undersampling)

## 5) Evaluation Metrics
Use classification metrics, especially for the churn class:
- Recall: very important because you want to catch as many real churners as possible
- Precision: helps control false alarms
- F1-score: balances precision and recall
- ROC-AUC: useful for overall ranking quality

Emphasize Recall for churn prediction, because missing a real churner can mean losing a customer.

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
