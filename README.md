# 🏦 Bank Loan Amount Prediction

**Regression model predicting how much loan a customer should be granted, using only pre-application customer attributes.**

> Part of a 5-model series on the same dataset (`loan_data.csv`), each answering a different business question. This repo covers **loan amount prediction** only. A combined README comparing all 5 models will be published separately once every model is completed.

**Dataset:** [Bank Loan Data — Kaggle](https://www.kaggle.com/datasets/udaymalviya/bank-loan-data) — 45,000 records, 14 columns

| Item | Detail |
|------|--------|
| **Target** | `loan_amnt` (USD) |
| **Problem type** | Regression |
| **Models** | Linear Regression (baseline) → Random Forest → Hyperparameter Tuning |
| **Stack** | Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Tableau |

---

## 📁 Repository Structure

```
bank-loan-amount-prediction/
│
├── bank_loan_amount_prediction.ipynb          # Main notebook — clean (no outputs)
├── bank_loan_amount_prediction_results.ipynb  # Notebook with all outputs & charts
├── bank_loan_amount_prediction.py             # Standalone Python script
├── loan_data.csv                              # Dataset (45,000 rows)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔁 ML Pipeline

```
Raw Data (CSV)
    │
    ▼
1. Data Loading & Inspection
    │
    ▼
2. Preprocessing
   ├── Drop post-approval leakage (loan_int_rate, loan_percent_income, loan_status)
   ├── Impute missing values (median) — no missing values found
   └── One-hot encode 5 categorical columns (done ONCE only)
    │
    ▼
3. Feature Engineering
   ├── Income_per_Emp_Year = person_income / (person_emp_exp + 1)
   └── Credit_History_per_Age = cb_person_cred_hist_length / (person_age + 1)
    │
    ▼
4. Train-Test Split (80/20, random_state=42)
   ├── Training set: 36,000 rows
   └── Test set    :  9,000 rows
    │
    ▼
5. StandardScaler — fit on TRAIN only, transform TEST separately
    │
    ▼
6. Model Training & Evaluation
   ├── Linear Regression (baseline)
   └── Random Forest Regressor (default)
    │
    ▼
7. Hyperparameter Tuning (RandomizedSearchCV, 3-fold CV)
    │
    ▼
8. Feature Importance (from tuned model only)
    │
    ▼
9. Tableau Export (test-set rows only — 9,000 rows)
```

---

## 📊 Results

### Target Variable — `loan_amnt`

| Stat | Value |
|------|-------|
| Mean | $9,583 |
| Median | $8,000 |
| Std | $6,315 |
| Min | $500 |
| Max | $35,000 |

### Test Set Performance (9,000 rows)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression (baseline) | $4,660 | $5,993 | 0.1077 |
| Random Forest (default) | $4,443 | $5,771 | 0.1727 |
| **Random Forest (tuned)** | **$4,382** | **$5,668** | **0.2019** |

### Cross-Validation R² (5-fold, training set only)

| Model | Mean R² | Std |
|-------|---------|-----|
| Linear Regression | 0.0833 | ±0.0157 |
| Random Forest | 0.1895 | ±0.0101 |

### Best Hyperparameters (RandomizedSearchCV)

```python
{
    'n_estimators':      200,
    'max_depth':         30,
    'min_samples_split': 2,
    'min_samples_leaf':  4,
    'max_features':      'sqrt'
}
```

---

## 🔍 Feature Importance (Tuned Random Forest)

| Rank | Feature | Importance | Note |
|------|---------|-----------|------|
| 1 | `person_income` | 0.3552 | Strongest predictor |
| 2 | `Income_per_Emp_Year` | 0.1644 | ✨ Engineered feature |
| 3 | `credit_score` | 0.1036 | |
| 4 | `Credit_History_per_Age` | 0.0713 | ✨ Engineered feature |
| 5 | `person_emp_exp` | 0.0574 | |
| 6 | `person_age` | 0.0555 | |
| 7 | `cb_person_cred_hist_length` | 0.0384 | |
| 8 | `previous_loan_defaults_on_file` | 0.0286 | |

> Both engineered features rank in the top 4 — confirming the value of feature engineering.

---

## 🛡️ Data Leakage Prevention

| Column | Why Excluded |
|--------|-------------|
| `loan_int_rate` | Set by bank **after** loan approval — not available at prediction time |
| `loan_percent_income` | = `loan_amnt / person_income` — directly uses the **target variable** |
| `loan_status` | Approval decision made **after** loan amount is determined |

Additional safeguards:
- `StandardScaler` fitted **only on `X_train`**
- `get_dummies` called **once only** in preprocessing
- Tableau export uses **`X_test.index`** — no training rows

---

## 📈 Tableau Export Files

| File | Content |
|------|---------|
| `bank_loan_tableau_export.csv` | Test-set rows (9,000) with `loan_amnt` (actual), `Predicted_Loan_Amount`, `Prediction_Error`, `Abs_Error` |
| `feature_importance_tableau.csv` | `Feature`, `Importance_Score` from the tuned model |
| `model_metrics_summary.csv` | `Model`, `MAE`, `RMSE`, `R²` for all 3 models — used for KPI cards |

Dashboard link: [Bank Loan Amount Prediction — Actual vs Predicted](https://public.tableau.com/app/profile/angela.nguyen6789/viz/BankLoanAmountPrediction/ActualvsPredicted)

---

## 🛠️ Setup & Usage

### Option A — Jupyter Notebook
```bash
git clone https://github.com/angelaadida/bank-loan-amount-prediction.git
cd bank-loan-amount-prediction
pip install -r requirements.txt
jupyter notebook bank_loan_amount_prediction.ipynb
```

### Option B — Python Script
```bash
python bank_loan_amount_prediction.py
```

> Make sure `loan_data.csv` is in the same folder.

---

## 🧠 Key Technical Decisions

| Decision | Why |
|----------|-----|
| Drop 3 leakage columns | Post-approval info — unavailable at prediction time |
| One-hot encode once | Multiple `get_dummies` calls create duplicate columns |
| Scale after split | Fitting scaler on full data leaks test statistics |
| Feature importance from tuned RF only | Multiple models give inconsistent results |
| Export test rows only | In-sample predictions inflate apparent accuracy |

---

## 📚 References

- [Dataset: Kaggle — udaymalviya/bank-loan-data](https://www.kaggle.com/datasets/udaymalviya/bank-loan-data)
- [Scikit-learn: RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Scikit-learn: RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
- [Tableau Public](https://public.tableau.com/)
