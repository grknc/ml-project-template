## Project Overview
Airline passenger satisfaction is important for retention, brand perception, and operational improvement. This project predicts whether a passenger is satisfied or neutral/dissatisfied using survey and service-related variables. The goal is to identify service quality drivers and provide actionable recommendations.

## Problem Definition
Target: predict customer satisfaction class (**satisfied** vs **neutral/dissatisfied**). This is a **binary classification** problem. The analysis should highlight which operational and service factors are most influential.

## Datasets
- **Name:** Airline Passenger Satisfaction
- **Source:** <https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction>
- Contains passenger profile fields and ratings for multiple service dimensions.
- Includes a satisfaction label for supervised classification.
- Suitable for feature importance analysis to identify service quality drivers.

## Recommended Models / Methods
- **Baseline:** Logistic Regression for a transparent benchmark.
- Decision Tree and Random Forest for non-linear relationships and interpretability.
- Gradient Boosting classifiers for stronger predictive performance on tabular data.
- Use feature importance analysis to convert model output into operational actions.

## Evaluation Metrics
- **ROC-AUC:** threshold-independent view of ranking performance.
- **F1-score:** balance between precision and recall.
- **Recall:** important to capture dissatisfied-risk passengers.
- **Confusion Matrix:** practical breakdown of prediction errors for operations teams.

## Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib / seaborn

## Expected Deliverables
- EDA notebook on satisfaction distribution and service quality factors.
- Feature engineering and preprocessing for classification modeling.
- At least two model experiments, including a Logistic Regression baseline.
- Evaluation and comparison using ROC-AUC, F1, recall, and confusion matrix.
- Short business interpretation with actionable operational recommendations.
