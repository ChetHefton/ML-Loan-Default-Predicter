# Credit Risk Analysis and Loan Default Prediction

This repository contains a complete machine learning pipeline designed to predict the probability of a borrower experiencing serious financial delinquency within two years. This project demonstrates data preprocessing, exploratory data analysis (EDA), and a comparative study of classification algorithms on highly imbalanced financial data.

## Business Problem

Predicting credit default is a critical task for financial institutions. The goal is to accurately identify high-risk borrowers (maximizing Recall) while maintaining a low false-positive rate to avoid denying credit to reliable customers.

## Technical Workflow

1. **Exploratory Data Analysis (EDA):** Visualized class imbalance and identified missing values in MonthlyIncome and NumberOfDependents.
2. **Data Preprocessing:** Implemented median imputation for missing data and clipped values at the 1st and 99th percentiles to mitigate the impact of financial outliers.
3. **Model Implementation:** Compared Logistic Regression (baseline), Random Forest (with class weighting), and Gradient Boosting.
4. **Evaluation:** Utilized AUC-ROC and F1-Score as primary metrics due to the significant class imbalance.

## Model Performance Discussion

The models were evaluated on a validation set of 30,000 samples. While overall accuracy remains high (~93-94%), the focus was on the minority class (defaults).

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1 (Class 1) | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9376 | 0.6125 | 0.1805 | 0.2789 | 0.8578 |
| **Random Forest** | 0.9322 | 0.4888 | **0.3162** | **0.3840** | 0.8528 |
| **Gradient Boosting** | **0.9380** | **0.6065** | 0.2060 | 0.3075 | **0.8680** |

### Analysis of Results:
- **Gradient Boosting** emerged as the strongest overall performer with an **AUC of 0.868**, indicating a superior ability to distinguish between defaulters and non-defaulters across various probability thresholds.
- **Random Forest** achieved the highest **Recall (0.316)** and **F1-Score (0.384)**. In a lending context, this suggests that the Random Forest model is the most effective at catching actual defaults, likely due to the implementation of balanced class weights which forces the model to prioritize the minority class.
- **Logistic Regression** provided a surprisingly competitive AUC (0.857), suggesting that the relationship between the features and default risk has significant linear components.

## Feature Importance

Analysis of the Gradient Boosting model reveals the primary drivers of default risk:


- **Number of Times 90 Days Late:** By far the most significant predictor. Past behavior is the strongest indicator of future delinquency.
- **Revolving Utilization of Unsecured Lines:** High utilization of credit limits relative to total available credit is the second most critical factor.
- **Historical Delinquency (30-59 and 60-89 days):** Short-term historical struggles are heavily weighted by the model, confirming that minor delinquency often cascades into serious delinquency.
- **Demographics:** Features like `age` and `NumberOfDependents` showed relatively low importance compared to actual credit utilization and payment history.

## How to run

### Option 1: Run locally
1. Create a virtual environment (recommended)
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   Open and run the notebook in Jupyter or VS Code.

### Option 2: Run in Google Colab

1 . The notebook can download the CSVs directly from GitHub using wget (already included in the notebook).

## Planned improvements (WIP)

- Better handling of class imbalance (threshold tuning, PR curves, class weights review, possibly SMOTE)

- Feature engineering and clearer justification for clipping/outlier rules

- Cross-validation instead of a single split

- More consistent preprocessing pipeline (e.g., Pipeline / ColumnTransformer)
