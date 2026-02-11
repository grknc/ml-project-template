# 22 Wine Quality Prediction

## 1) Project Overview
This project predicts wine quality from physicochemical features such as acidity, sugar, and alcohol. You can solve it as classification (quality groups) or regression (exact quality score). The objective is to understand which features influence wine quality the most.

## 2) Problem Definition
- **What is predicted or analyzed:** Wine quality.
- **Target:** Quality label category (classification) or quality score (regression).
- **Problem type:** Classification or regression depending on project setup.
- **Key note:** Preprocessing is important, especially scaling numeric features and checking feature importance.

## 3) Suggested Datasets
- **Name:** Wine Quality
- **Dataset Source:** <https://www.kaggle.com/datasets/yasserh/wine-quality-dataset>
- Features usually include fixed acidity, volatile acidity, citric acid, pH, sulphates, and alcohol.

## 4) Recommended Models / Methods
- **Baseline:** Logistic Regression (for classification) or simple linear baseline (for regression framing).
- Random Forest.
- Gradient Boosting.
- Compare feature importance methods to explain key quality drivers.

## 5) Evaluation Metrics
- **If classification:** F1-score and accuracy.
- **If regression:** MAE and RMSE.
- Use confusion matrix (classification) or residual analysis (regression) for additional insight.

## 6) Tools & Libraries
- Python.
- pandas, numpy.
- scikit-learn.
- matplotlib / seaborn.

## 7) Expected Deliverables
- Clear problem framing: classification vs regression and why.
- Preprocessing summary with scaling decisions.
- Model comparison table with selected metrics.
- Feature importance interpretation and final recommendation.

## 8) Common Pitfalls & Tips
- Do not skip feature scaling for linear models.
- Decide early whether quality is categories or numeric score.
- Watch out for class imbalance in classification setup.
- Avoid overfitting with too many tuned parameters.
- Validate feature importance with more than one method.
