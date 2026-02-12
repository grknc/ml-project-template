## Project Overview
Credit card fraud detection helps financial teams reduce losses and protect customers. This project focuses on identifying fraudulent transactions in a highly imbalanced setting where fraud cases are very rare. The goal is to build reliable models and decision thresholds that support operational fraud review.

## Problem Definition
Target: classify each transaction as fraud or non-fraud. This is an **imbalanced binary classification** problem. The project should explicitly manage the precision/recall trade-off because missing fraud (false negatives) is often costly.

## Datasets
- **Name:** Credit Card Fraud Detection
- **Source:** <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
- Contains transaction-level features prepared for fraud modeling.
- Includes a binary fraud label suitable for supervised classification.
- Represents an extreme class imbalance scenario that requires careful validation and threshold tuning.

## Recommended Models / Methods
- **Baseline:** Logistic Regression with class weighting.
- Tree-based models such as Random Forest and Gradient Boosting.
- Additional imbalance-aware strategies: resampling, class weights, and calibrated threshold tuning.
- Compare model performance under different decision thresholds for operational use.

## Evaluation Metrics
- **Precision:** measures how many flagged transactions are truly fraud.
- **Recall:** measures how much fraud is captured; critical when false negatives are expensive.
- **F1-score:** balances precision and recall for overall decision quality.
- **PR-AUC (optional):** often informative in extreme imbalance.
- **ROC-AUC:** useful as a general ranking metric across thresholds.

## Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib / seaborn

## Expected Deliverables
- EDA notebook highlighting imbalance and transaction behavior.
- Feature engineering and preprocessing for fraud signals.
- At least two model experiments, including a Logistic Regression baseline.
- Evaluation and comparison with threshold analysis.
- Short business interpretation and recommendations for fraud operations.
