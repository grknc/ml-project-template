## Project Overview
Customer segmentation helps companies group customers with similar behavior and value patterns. These segments support targeted marketing, personalized offers, and better customer management strategies. The goal is to find meaningful groups that can guide business actions.

## Problem Definition
The project analyzes customer patterns to discover distinct groups.
This is an **unsupervised learning** problem, usually solved with clustering.

## Datasets
- **Dataset name:** Customer Segmentation Dataset
- **Source:** https://www.kaggle.com/datasets/yasserh/customer-segmentation-dataset/data
- **Description:** The dataset contains customer attributes such as demographics, spending behavior, and engagement-related variables used for clustering.

## Recommended Models / Methods
- **Baseline:** Rule-based grouping using one or two key variables (for example, spending level).
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- PCA for dimensionality reduction before clustering (optional)

## Evaluation Metrics
- **Silhouette Score:** Measures how well each point fits its cluster.
- **Davies-Bouldin Index:** Evaluates cluster compactness and separation.
- **Calinski-Harabasz Score:** Compares between-cluster and within-cluster dispersion.
- **Business interpretability:** Ensures segments are useful for real decisions.

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
