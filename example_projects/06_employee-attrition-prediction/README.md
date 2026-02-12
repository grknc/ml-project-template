## Project Overview
Employee attrition is a key HR problem because losing experienced employees increases hiring cost, training time, and team disruption. This project aims to predict whether an employee will leave the company so HR teams can act early. The main goal is to combine solid prediction with clear explanations that support real retention actions.

## Problem Definition
Target: predict **Attrition (Yes/No)** for each employee record. This is a **binary classification** problem. The project should also provide interpretable findings so HR teams can understand the drivers of attrition and plan interventions.

## Datasets
- **Name:** HR Analytics Attrition Dataset (IBM HR Analytics)
- **Source:** <https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset>
- Contains employee demographics, job role, compensation, work environment, and satisfaction-related fields.
- Includes a binary attrition label that supports supervised learning.
- Can show class imbalance, so label distribution should be checked before modeling.

## Recommended Models / Methods
- **Baseline:** Logistic Regression for a simple and interpretable baseline.
- Decision Tree and Random Forest for non-linear patterns and feature importance.
- Gradient Boosting methods (for example, XGBoost/LightGBM if available) for stronger tabular performance.
- Use class weights, resampling, or threshold tuning when attrition cases are underrepresented.
- Add model interpretation (feature importance or similar) to produce actionable HR insights.

## Evaluation Metrics
- **Recall (attrition class):** important to catch as many at-risk employees as possible.
- **F1-score:** useful balance between precision and recall in imbalanced settings.
- **ROC-AUC:** evaluates ranking quality across decision thresholds.
- Accuracy can be reported, but it should not be the main metric if classes are imbalanced.

## Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib / seaborn

## Expected Deliverables
- EDA notebook focused on attrition patterns and risk factors.
- Customer-level feature engineering equivalent for HR data (role, tenure, compensation, satisfaction signals).
- At least two model experiments, including a Logistic Regression baseline.
- Evaluation and comparison using recall, F1, and ROC-AUC.
- Short business interpretation with practical HR recommendations.
