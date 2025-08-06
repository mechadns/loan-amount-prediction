# Loan Approval Prediction

Predicting loan approval based on applicant information using **machine learning**.  
This project explores multiple models, performs **EDA**, applies **feature engineering**, and tunes hyperparameters to achieve strong predictive performance.

---

## Project Overview

The goal is to build a **classification model** that predicts whether a loan application will be approved (`Y`) or not approved (`N`) based on applicant attributes like income, education, employment status, and loan details.

We used:
- **EDA** to understand patterns in the dataset
- **Data preprocessing** to handle missing values and encode categorical variables
- **Model evaluation** to compare multiple algorithms
- **Hyperparameter tuning** for improved performance
- **Final tuned model** pipeline for deployment

---

## 📂 Dataset

Dataset: [Loan Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset?select=train_u6lujuX_CVtuZ9i.csv)

| Column | Description |
|--------|-------------|
| Loan_ID | Unique loan identifier |
| Gender | Male / Female |
| Married | Applicant married (Y/N) |
| Dependents | Number of dependents |
| Education | Graduate / Not Graduate |
| Self_Employed | Self-employed (Y/N) |
| ApplicantIncome | Applicant's income |
| CoapplicantIncome | Co-applicant's income |
| LoanAmount | Loan amount in thousands |
| Loan_Amount_Term | Term of loan in months |
| Credit_History | Credit history meets guidelines (1/0) |
| Property_Area | Urban / Semiurban / Rural |
| Loan_Status | Loan approved (Y/N) – **target variable**

---

## Exploratory Data Analysis (EDA)

Some key findings:
- Applicants with a **credit history** are far more likely to get loan approval.
- Higher applicant income **alone** does not guarantee approval; credit history is more influential.
- Semiurban property area had slightly higher approval rates.
- Missing values in `LoanAmount` and `Credit_History` were filled using median and mode imputation.

---

## Model Development

We evaluated multiple models:

| Model | Accuracy | F1-score | ROC-AUC |
|-------|----------|----------|---------|
| **Random Forest (Tuned)** | **0.8143** | **0.8779** | **0.7814** |
| CatBoost (Tuned) | 0.8144 | 0.8788 | 0.7782 |
| XGBoost (Tuned) | 0.8078 | 0.8743 | 0.7744 |
| Logistic Regression | 0.8046 | 0.8729 | 0.7495 |
| LightGBM | 0.7752 | 0.8429 | 0.7587 |
| Gradient Boosting | 0.7850 | 0.8533 | 0.7527 |

---

## Final Model

We selected the **Tuned Random Forest Classifier** as the final model because:
- Consistent high **F1-score** and **Accuracy**
- Robust to overfitting
- Interpretable feature importances

---

## Final Evaluation

On cross-validation:

- **Accuracy:** 0.8143  
- **Precision:** 0.8779  
- **Recall:** 0.7814  
- **F1-score:** 0.8779  
- **ROC-AUC:** 0.7814

---

## Submission

The final pipeline:
- **Preprocessing:** Missing value imputation + One-hot encoding
- **Classifier:** Tuned Random Forest
- Saved with `joblib` for reproducibility

Sample predictions file: [`results.csv`](results.csv)

```csv
Loan_ID,Loan_Status
LP001015,Y
LP001022,N
LP001031,Y
...
