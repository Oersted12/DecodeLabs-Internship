import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# =============================================================================
# CONFIGURATION  ← only change this line
# =============================================================================

DATASET_PATH = "C:\\Users\\chris\\OneDrive - Sri Lanka Institute of Information Technology\\DecodeLabs\\Task 2\\Project 2\\data\\Dataset for Data Analytics.xlsx"
RANDOM_STATE = 42


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print("=" * 65)
print("STEP 1: Loading Data")
print("=" * 65)

df = pd.read_excel(DATASET_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:\n{df.head(3)}\n")


# =============================================================================
# STEP 2: ENGINEER FRAUD LABEL
# =============================================================================

print("=" * 65)
print("STEP 2: Engineering Fraud Label")
print("=" * 65)

high_value_threshold = df['TotalPrice'].quantile(0.75)
print(f"High-value threshold (75th percentile): ${high_value_threshold:.2f}")

df['isFraud'] = (
    (df['OrderStatus'].isin(['Cancelled', 'Returned'])) &
    (df['CouponCode'].isna()) &
    (df['TotalPrice'] > high_value_threshold)
).astype(int)

fraud_count = df['isFraud'].sum()
legit_count = len(df) - fraud_count
fraud_rate  = fraud_count / len(df) * 100

print(f"\nFraudulent transactions : {fraud_count}  ({fraud_rate:.2f}%)")
print(f"Legitimate transactions : {legit_count}  ({100 - fraud_rate:.2f}%)")
print(f"\n>>> Class imbalance ratio 1:{legit_count // fraud_count} (legit:fraud)")
print(">>> This is why we NEED SMOTE — accuracy alone would be misleading!\n")


# =============================================================================
# STEP 3: FEATURE ENGINEERING
# =============================================================================

print("=" * 65)
print("STEP 3: Feature Engineering")
print("=" * 65)

df['Date']       = pd.to_datetime(df['Date'])
df['DayOfWeek']  = df['Date'].dt.dayofweek
df['Month']      = df['Date'].dt.month
df['IsWeekend']  = (df['DayOfWeek'] >= 5).astype(int)

df['CouponUsed']    = df['CouponCode'].notna().astype(int)
coupon_map          = {'FREESHIP': 1, 'SAVE10': 2, 'WINTER15': 3}
df['CouponEncoded'] = df['CouponCode'].map(coupon_map).fillna(0).astype(int)

df['PriceRatio'] = df['UnitPrice'] / df['UnitPrice'].mean()

le = LabelEncoder()
df['Product_enc']        = le.fit_transform(df['Product'])
df['PaymentMethod_enc']  = le.fit_transform(df['PaymentMethod'])
df['ReferralSource_enc'] = le.fit_transform(df['ReferralSource'])
df['OrderStatus_enc']    = le.fit_transform(df['OrderStatus'])

FEATURES = [
    'Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice',
    'DayOfWeek', 'Month', 'IsWeekend',
    'CouponUsed', 'CouponEncoded', 'PriceRatio',
    'Product_enc', 'PaymentMethod_enc', 'ReferralSource_enc', 'OrderStatus_enc'
]

X = df[FEATURES]
y = df['isFraud']

print(f"Features used ({len(FEATURES)}): {FEATURES}")
print(f"\nX shape: {X.shape}")
print(f"y distribution:\n{y.value_counts()}\n")


# =============================================================================
# STEP 4: TRAIN / TEST SPLIT
# =============================================================================

print("=" * 65)
print("STEP 4: Train/Test Split (80/20, stratified)")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set : {X_train.shape[0]} rows")
print(f"Test set     : {X_test.shape[0]} rows")
print(f"\nTraining fraud distribution:\n{y_train.value_counts()}")
print(f"\nTest fraud distribution:\n{y_test.value_counts()}\n")


# =============================================================================
# STEP 5: BUILD PIPELINES
# =============================================================================

print("=" * 65)
print("STEP 5: Building imblearn Pipelines")
print("=" * 65)

lr_pipeline = ImbPipeline(steps=[
    ('scaler',     StandardScaler()),
    ('smote',      SMOTE(random_state=RANDOM_STATE)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])

rf_pipeline = ImbPipeline(steps=[
    ('smote',      SMOTE(random_state=RANDOM_STATE)),
    ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
])

print("Pipeline A (Logistic Regression): StandardScaler → SMOTE → LR")
print("Pipeline B (Random Forest)      : SMOTE → RF")
print(">>> Both pipelines are leak-free: SMOTE only sees training folds\n")


# =============================================================================
# STEP 6: HYPERPARAMETER TUNING WITH GridSearchCV
# =============================================================================

print("=" * 65)
print("STEP 6: Hyperparameter Tuning (GridSearchCV, 5-fold CV)")
print("=" * 65)

lr_param_grid = {
    'smote__k_neighbors': [3, 5, 7],
    'classifier__C':      [0.01, 0.1, 1.0]
}

print("Tuning Logistic Regression pipeline...")
lr_grid_search = GridSearchCV(
    lr_pipeline, lr_param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=0
)
lr_grid_search.fit(X_train, y_train)
print(f"  Best params   : {lr_grid_search.best_params_}")
print(f"  Best CV Recall: {lr_grid_search.best_score_:.4f}")

rf_param_grid = {
    'smote__k_neighbors':       [3, 5, 7],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth':    [10, 20, None]
}

print("\nTuning Random Forest pipeline...")
rf_grid_search = GridSearchCV(
    rf_pipeline, rf_param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=0
)
rf_grid_search.fit(X_train, y_train)
print(f"  Best params   : {rf_grid_search.best_params_}")
print(f"  Best CV Recall: {rf_grid_search.best_score_:.4f}\n")


# =============================================================================
# STEP 7: EVALUATE ON TEST SET
# =============================================================================

print("=" * 65)
print("STEP 7: Final Evaluation on Test Set")
print("=" * 65)

def evaluate_model(name, grid_search, X_test, y_test):
    best_model = grid_search.best_estimator_
    y_pred     = best_model.predict(X_test)
    y_proba    = best_model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_proba)

    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  Precision : {precision:.4f}  (fraud flags that are correct)")
    print(f"  Recall    : {recall:.4f}  (actual fraud cases caught)")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}  (target: >= 0.85)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Legitimate', 'Fraud'],
                                zero_division=0))
    return y_pred, y_proba, precision, recall, f1, roc_auc

lr_pred, lr_proba, lr_p, lr_r, lr_f1, lr_auc = evaluate_model(
    "Logistic Regression (best params)", lr_grid_search, X_test, y_test
)
rf_pred, rf_proba, rf_p, rf_r, rf_f1, rf_auc = evaluate_model(
    "Random Forest (best params)", rf_grid_search, X_test, y_test
)


# =============================================================================
# STEP 8: VISUAL OUTPUTS
# =============================================================================

print("\n" + "=" * 65)
print("STEP 8: Generating Plots")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("DecodeLabs Project 2 — Fraud Detection Results", fontsize=15, fontweight='bold')

# Plot 1: Class Distribution
ax = axes[0, 0]
counts = [legit_count, fraud_count]
colors = ['#4C9BE8', '#E85C5C']
bars = ax.bar(['Legitimate (0)', 'Fraud (1)'], counts, color=colors, edgecolor='black')
ax.set_title("Class Distribution (Original)", fontsize=12, fontweight='bold')
ax.set_ylabel("Count")
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(count), ha='center', va='bottom', fontweight='bold')
ax.set_ylim(0, max(counts) * 1.15)

# Plot 2: Confusion Matrix LR
ax = axes[0, 1]
ConfusionMatrixDisplay(confusion_matrix(y_test, lr_pred),
                       display_labels=['Legitimate', 'Fraud']).plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title("Confusion Matrix\nLogistic Regression", fontsize=12, fontweight='bold')

# Plot 3: Confusion Matrix RF
ax = axes[0, 2]
ConfusionMatrixDisplay(confusion_matrix(y_test, rf_pred),
                       display_labels=['Legitimate', 'Fraud']).plot(ax=ax, colorbar=False, cmap='Greens')
ax.set_title("Confusion Matrix\nRandom Forest", fontsize=12, fontweight='bold')

# Plot 4: ROC Curves
ax = axes[1, 0]
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
ax.plot(fpr_lr, tpr_lr, color='#4C9BE8', lw=2, label=f'Logistic Regression (AUC = {lr_auc:.3f})')
ax.plot(fpr_rf, tpr_rf, color='#27AE60', lw=2, label=f'Random Forest (AUC = {rf_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("ROC Curves", fontsize=12, fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

# Plot 5: Metrics Comparison
ax = axes[1, 1]
metrics   = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x_pos     = np.arange(len(metrics))
width     = 0.35
bars1 = ax.bar(x_pos - width/2, [lr_p, lr_r, lr_f1, lr_auc], width,
               label='Logistic Regression', color='#4C9BE8', edgecolor='black')
bars2 = ax.bar(x_pos + width/2, [rf_p, rf_r, rf_f1, rf_auc], width,
               label='Random Forest', color='#27AE60', edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.15)
ax.set_title("Model Metrics Comparison\n(No Accuracy — by design)", fontsize=12, fontweight='bold')
ax.set_ylabel("Score")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ax.axhline(y=0.85, color='red', linestyle='--', linewidth=1)
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

# Plot 6: Feature Importance
ax = axes[1, 2]
best_rf     = rf_grid_search.best_estimator_.named_steps['classifier']
feat_imp    = pd.Series(best_rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
colors_imp  = ['#E85C5C' if v > feat_imp.quantile(0.75) else '#4C9BE8' for v in feat_imp]
feat_imp.plot(kind='barh', ax=ax, color=colors_imp, edgecolor='black')
ax.set_title("Feature Importance\n(Random Forest)", fontsize=12, fontweight='bold')
ax.set_xlabel("Importance Score")

plt.subplots_adjust(hspace=0.8, wspace=0.4)
plt.tight_layout(pad=3.0)

import os
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/fraud_detection_results.png", dpi=150, bbox_inches='tight')
print("  Saved → outputs/fraud_detection_results.png")
plt.show()


# =============================================================================
# STEP 9: FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("STEP 9: Final Summary")
print("=" * 65)

winner = "Random Forest" if rf_auc > lr_auc else "Logistic Regression"

print("Pipeline complete. Check outputs/fraud_detection_results.png")
