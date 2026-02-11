# Loan Default Prediction (ML Project)

This repo originated as a course project but has been expanded upon. It contains a machine learning workflow for predicting whether a borrower will become seriously delinquent within 2 years (`SeriousDlqin2yrs`) using tabular credit features.

The work is currently in a notebook-style format and uses the provided training/test CSV files.

---

## What’s in here

- `loan_default_prediction.ipynb` (or similarly named notebook)  
  - Loads data
  - Basic EDA (missing values, class imbalance, correlations, outliers)
  - Preprocessing (median imputation, clipping outliers, scaling where needed)
  - Trains and compares models:
    - Logistic Regression (baseline)
    - Random Forest
    - Gradient Boosting
  - Evaluates with:
    - Confusion matrix
    - Precision/Recall/F1 (especially for class 1)
    - ROC curve + AUC
  - Generates test-set probabilities and writes a CSV (`test_predictions.csv`)

- `cs-training.csv` / `cs-test.csv`  
  Dataset files used by the notebook.

---

## Notes on Results

This is an imbalanced classification problem (far fewer positives than negatives).  
Accuracy is not the main metric to watch. I focused more on:

- Recall / F1 for the positive class (default)
- ROC-AUC

The notebook includes a small comparison table across models.

---

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
