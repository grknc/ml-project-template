# 06 Employee Attrition Prediction

## 1) Project Overview
This project estimates whether an employee is likely to leave the company. The goal is to help HR teams identify retention risks early and design better support plans. Attrition prediction can improve workforce stability and reduce hiring costs. The project should balance prediction quality with clear explanations for decision-makers.

## 2) Problem Definition
- **What is predicted or analyzed:** The likelihood of employee attrition.
- **Target:** Binary label (attrition: yes/no).
- **Typical stakeholders / business value:** HR leaders and people managers use results for retention programs, hiring planning, and policy improvements.
- **Important note:** Interpretability is key so HR can understand and trust model signals.

## 3) Suggested Datasets
- **IBM HR Analytics Employee Attrition & Performance (Kaggle):** Popular benchmark for attrition classification.
- **HR Employee Attrition (Kaggle variants):** Multiple versions with role, satisfaction, and compensation features.
- **Employee datasets from UCI-style repositories:** Useful for practicing binary classification with tabular HR data.

## 4) Recommended Models / Methods
- **Baseline approach:** Logistic Regression as a simple and interpretable baseline.
- Decision Tree / Random Forest for non-linear patterns.
- Gradient Boosting, XGBoost, or LightGBM for stronger predictive performance.
- Use feature importance and simple explainability tools for HR insights.
- Handle class imbalance with class weights, resampling, or threshold tuning.

## 5) Evaluation Metrics
- **Recall (attrition class):** Important to catch as many at-risk employees as possible.
- **F1-score:** Balances recall and precision when classes are uneven.
- **ROC-AUC:** Measures ranking quality across different thresholds.
- Class imbalance is common, so accuracy alone can be misleading.

## 6) Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib / seaborn
- xgboost / lightgbm (optional)

## 7) Expected Deliverables
- EDA notebook with attrition patterns by role, tenure, and satisfaction.
- Feature engineering for tenure, compensation bands, or department signals.
- At least 2 experiments (for example, Logistic Regression vs boosted trees).
- Model evaluation and comparison with focus on attrition recall.
- Business insights / conclusions for practical HR actions.
- Clear README + reproducible steps.
