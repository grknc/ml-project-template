# 10 Customer Satisfaction Prediction

## 1) Project Overview
This project predicts customer satisfaction from survey answers and service details. You can use data from airlines, hotels, or e-commerce platforms. The main goal is to help teams understand what drives satisfaction and what should be improved first.

## 2) Problem Definition
- **What is predicted or analyzed:** Customer satisfaction level after a product or service experience.
- **Target:** Either a class label (satisfied / not satisfied) or a numeric score.
- **Typical stakeholders / business value:** Customer experience teams, operations teams, and product managers use the results to improve service quality.
- **Common challenges:** Missing survey values, biased responses, and class imbalance.

## 3) Suggested Datasets
- **Airline Passenger Satisfaction (Kaggle):** Survey data with service quality features.
  
  https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
- **Hotel Review / Guest Satisfaction datasets (Kaggle/public):** Ratings and service-related variables.
- **E-commerce customer feedback datasets:** Order, delivery, and support experience features.

## 4) Recommended Models / Methods
- **Baseline approach:** Logistic Regression (classification) or Linear Regression (if using score prediction).
- Random Forest.
- Gradient Boosting.
- Include feature importance or model explainability to produce actionable insights.

## 5) Evaluation Metrics
- For **classification**: Accuracy, Recall, F1-score, ROC-AUC.
- For **regression**: RMSE (and optionally MAE).
- Compare models and explain which one gives better practical insights.

## 6) Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib / seaborn
- xgboost / lightgbm (optional)

## 7) Expected Deliverables
- Data understanding and cleaning summary.
- Problem framing (classification or regression) with clear target definition.
- At least 2–3 model experiments and comparison table.
- Key drivers of satisfaction and recommended actions.
- Final report and reproducible README steps.
