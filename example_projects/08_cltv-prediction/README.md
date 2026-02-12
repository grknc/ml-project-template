## Project Overview
Customer Lifetime Value (CLTV) prediction helps businesses allocate marketing and retention budgets more effectively. This project estimates future customer value using transaction history and customer-level features. The goal is to support segmentation, campaign prioritization, and better revenue planning.

## Problem Definition
Target: predict future customer value over a defined horizon. This is mainly a **regression** task, with an optional **probabilistic CLTV** perspective for customer behavior modeling. The project should provide interpretable outputs and segment-level business insights.

## Datasets
- **Name:** Online Retail II (UCI)
- **Source:** <https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci>
- Contains transaction-level retail data including invoice, quantity, price, and customer identifiers.
- Enables creation of customer-level features such as recency, frequency, and monetary value.
- Supports CLTV modeling through aggregated features and segment analysis.

## Recommended Models / Methods
- **Baseline:** simple historical average spend or linear regression baseline.
- Regression models such as Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor.
- Build robust customer-level features from transaction data before training.
- Optionally mention probabilistic CLTV methods conceptually (BG/NBD + Gamma-Gamma).
- Compare segment-level patterns to validate business usefulness.

## Evaluation Metrics
- **MAE:** clear average absolute error for business interpretation.
- **RMSE:** gives higher penalty to large prediction errors.
- Segment-level evaluation (for example, low/medium/high value groups) to check practical decision quality.
- Use metric comparison together with segment insights, not only a single score.

## Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib / seaborn

## Expected Deliverables
- EDA notebook on transaction behavior and customer segments.
- Customer-level feature engineering from transactional records.
- At least two model/method experiments, including a baseline.
- Evaluation and comparison using MAE/RMSE plus segment-level checks.
- Short business interpretation with CLTV-based recommendations.
