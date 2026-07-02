"""
Bank Loan Amount Prediction
Model 1 — Regression Pipeline
Dataset: loan_data.csv (Kaggle: udaymalviya/bank-loan-data)
Target: loan_amnt (USD)

Author: Angela Nguyen Hao
GitHub: https://github.com/angelaadida
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set(style='whitegrid')

# ─────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. LOAD DATA")
print("=" * 60)

df = pd.read_csv('loan_data.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ─────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. PREPROCESSING")
print("=" * 60)

# Drop post-approval leakage columns
# loan_int_rate      → set AFTER loan approval
# loan_percent_income → = loan_amnt / person_income (uses target)
# loan_status        → approval decision AFTER loan amount set
df = df.drop(columns=['loan_int_rate', 'loan_percent_income', 'loan_status'])
print(f"After dropping leakage columns: {df.shape}")

# Check missing values
missing = df.isnull().sum()
print(f"Missing values: {missing.sum()}")

# Impute if needed
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# One-hot encode categorical columns — DONE ONCE ONLY
cat_cols = ['person_gender', 'person_education', 'person_home_ownership',
            'loan_intent', 'previous_loan_defaults_on_file']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print(f"After encoding: {df.shape}")

obj_remaining = df.select_dtypes(include='object').columns.tolist()
assert len(obj_remaining) == 0, f"Object columns remain: {obj_remaining}"
print("✅ All columns numeric")

# ─────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. FEATURE ENGINEERING")
print("=" * 60)

# Income per year of employment — financial stability proxy
df['Income_per_Emp_Year'] = df['person_income'] / (df['person_emp_exp'] + 1)

# Credit history density per year of age
df['Credit_History_per_Age'] = df['cb_person_cred_hist_length'] / (df['person_age'] + 1)

print("New features: Income_per_Emp_Year, Credit_History_per_Age")

# ─────────────────────────────────────────────────────────────────
# 4. TRAIN-TEST SPLIT & SCALING
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. TRAIN-TEST SPLIT & SCALING")
print("=" * 60)

X = df.drop(columns=['loan_amnt'])
y = df['loan_amnt']

# Split BEFORE scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")
assert len(set(X_train.index) & set(X_test.index)) == 0
print("✅ No index overlap")

# Scale AFTER split — fit ONLY on X_train
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X.columns, index=X_test.index
)
print("✅ Scaling complete — fit on X_train only")

# ─────────────────────────────────────────────────────────────────
# 5. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. MODEL TRAINING & EVALUATION")
print("=" * 60)

def evaluate_model(y_true, y_pred, model_name):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    print(f"\n📊 {model_name}:")
    print(f"   MAE  : {mae:>12,.2f}")
    print(f"   MSE  : {mse:>12,.2f}")
    print(f"   RMSE : {rmse:>12,.2f}")
    print(f"   R²   : {r2:>12.4f}")
    print("-" * 45)
    return {"Model": model_name, "MAE": round(mae,2),
            "RMSE": round(rmse,2), "R²": round(r2,4)}

# Linear Regression (baseline)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Random Forest (default)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

results = []
results.append(evaluate_model(y_test, y_pred_lr, "Linear Regression"))
results.append(evaluate_model(y_test, y_pred_rf, "Random Forest (default)"))

# Cross-validation on training set only
cv_lr = cross_val_score(LinearRegression(), X_train_scaled, y_train, cv=5, scoring='r2')
cv_rf = cross_val_score(
    RandomForestRegressor(100, random_state=42, n_jobs=-1),
    X_train_scaled, y_train, cv=5, scoring='r2'
)
print(f"\nCross-Validation R² (5-fold):")
print(f"  Linear Regression : {cv_lr.mean():.4f} ±{cv_lr.std():.4f}")
print(f"  Random Forest     : {cv_rf.mean():.4f} ±{cv_rf.std():.4f}")

# ─────────────────────────────────────────────────────────────────
# 6. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. HYPERPARAMETER TUNING")
print("=" * 60)

param_dist = {
    'n_estimators':      [50, 100, 200],
    'max_depth':         [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2'],
}

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=10, cv=3,
    scoring='neg_mean_absolute_error',
    random_state=42, n_jobs=-1
)
rf_search.fit(X_train_scaled, y_train)
rf_tuned = rf_search.best_estimator_
y_pred_tuned = rf_tuned.predict(X_test_scaled)

print(f"Best params: {rf_search.best_params_}")
result_tuned = evaluate_model(y_test, y_pred_tuned, "Random Forest (Tuned)")
results.append(result_tuned)

results_df = pd.DataFrame(results).set_index("Model")
print("\n📊 Final Model Comparison:")
print(results_df.to_string())

# ─────────────────────────────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. FEATURE IMPORTANCE")
print("=" * 60)

feature_importance = pd.Series(
    rf_tuned.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("Top 10 Features:")
print(feature_importance.head(10).round(4).to_string())

# ─────────────────────────────────────────────────────────────────
# 8. TABLEAU EXPORT — test rows only
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. TABLEAU EXPORT")
print("=" * 60)

df_raw = pd.read_csv('loan_data.csv')
df_export = df_raw.loc[X_test.index].copy()
df_export['Predicted_Loan_Amount'] = y_pred_tuned.round(0).astype(int)
df_export['Prediction_Error']      = df_export['loan_amnt'] - df_export['Predicted_Loan_Amount']
df_export['Abs_Error']             = df_export['Prediction_Error'].abs()
df_export['Split']                 = 'Test'
df_export.to_csv('bank_loan_tableau_export.csv', index=False)
print(f"✅ Exported: bank_loan_tableau_export.csv ({len(df_export):,} rows)")

importance_df = pd.DataFrame({
    'Feature': feature_importance.index,
    'Importance_Score': feature_importance.values
})
importance_df.to_csv('feature_importance_tableau.csv', index=False)
print("✅ Exported: feature_importance_tableau.csv")

# Model comparison metrics — for KPI cards / bar chart in Tableau
results_df.reset_index().to_csv('model_metrics_summary.csv', index=False)
print("✅ Exported: model_metrics_summary.csv")

# ─────────────────────────────────────────────────────────────────
# 9. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. SAVING VISUALIZATIONS")
print("=" * 60)

# Model comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['tomato', 'steelblue', 'seagreen']
for ax, metric in zip(axes, ['MAE', 'RMSE', 'R²']):
    vals = results_df[metric]
    bars = ax.bar(vals.index, vals.values, color=colors[:len(vals)],
                  edgecolor='black', width=0.5)
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.set_ylabel(metric)
    for bar, v in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.01 if bar.get_height() >= 0 else bar.get_height() * 1.05,
                f'{v:,.2f}', ha='center', fontsize=9)
    ax.tick_params(axis='x', rotation=20)
plt.suptitle('Model Comparison — Test Set', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: model_comparison.png")

# Feature importance
plt.figure(figsize=(12, 7))
feature_importance.head(15).plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Top 15 Feature Importance — Random Forest (Tuned)', fontsize=13, fontweight='bold')
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: feature_importance.png")

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_tuned, alpha=0.3, color='steelblue', s=5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Loan Amount (USD)')
plt.ylabel('Predicted Loan Amount (USD)')
plt.title('Random Forest (Tuned): Actual vs Predicted', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: actual_vs_predicted.png")

# Residual plot
residuals = y_test - y_pred_tuned
plt.figure(figsize=(8, 5))
plt.scatter(y_pred_tuned, residuals, alpha=0.3, color='steelblue', s=5)
plt.axhline(0, color='red', linewidth=1.5, linestyle='--')
plt.xlabel('Predicted Loan Amount (USD)')
plt.ylabel('Residual (Actual − Predicted)')
plt.title('Residual Plot — Random Forest (Tuned)', fontweight='bold')
plt.tight_layout()
plt.savefig('residual_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: residual_plot.png")

print("\n" + "=" * 60)
print("✅ ALL DONE")
print("=" * 60)
