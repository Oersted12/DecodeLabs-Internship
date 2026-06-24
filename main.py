import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import pandera.pandas as pa
from pandera import Column, DataFrameSchema, Check
import os

os.makedirs("output", exist_ok=True)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
df = pd.read_excel("data/Dataset for Data Analytics.xlsx")
print("=== RAW DATA ===")
print(f"Shape: {df.shape}")
print(df.head())


# ============================================================
# STEP 2: MISSING VALUE ANALYSIS
# ============================================================
print("\n=== MISSING VALUE ANALYSIS ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_report = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct.round(2)
})
print(missing_report[missing_report["Missing Count"] > 0])

# Decision Matrix (from the DecodeLabs kit):
# < 5%   → Drop rows
# 5–20%  → Statistical imputation (median / group-wise)
# > 20%  → KNN imputation
#
# CouponCode is missing 25.75% → BUT it's categorical, not numeric.
# KNN doesn't apply to categorical directly.
# Best approach: fill with "NO_COUPON" — it's a meaningful category.

df["CouponCode"] = df["CouponCode"].fillna("NO_COUPON")
print("\nCouponCode after imputation:")
print(df["CouponCode"].value_counts())


# ============================================================
# STEP 3: OUTLIER DETECTION & NEUTRALIZATION (IQR)
# ============================================================
print("\n=== OUTLIER DETECTION (IQR) ===")

numeric_cols = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers | Bounds [{lower:.2f}, {upper:.2f}]")

    # Winsorization — cap values at boundary instead of dropping rows
    df[col] = np.clip(df[col], lower, upper)

print("\nAfter Winsorization:")
print(df[numeric_cols].describe().round(2))


# ============================================================
# STEP 4: FEATURE ENGINEERING (3 new features)
# ============================================================
print("\n=== FEATURE ENGINEERING ===")

# Feature 1: Revenue per Item
# How much revenue does each item in the cart generate on average?
df["RevenuePerItem"] = df["TotalPrice"] / df["ItemsInCart"]

# Feature 2: Has Coupon (binary flag)
# Did this order use a real coupon code?
df["HasCoupon"] = (df["CouponCode"] != "NO_COUPON").astype(int)

# Feature 3: Order Month
# Extract month from date — useful for seasonal trend analysis
df["OrderMonth"] = df["Date"].dt.month

# Feature 4 (bonus): Price Category
# Bin TotalPrice into Low / Medium / High segments
df["PriceCategory"] = pd.cut(
    df["TotalPrice"],
    bins=[0, 400, 1200, 9999],
    labels=["Low", "Medium", "High"]
)

print("New features added:")
print(df[["TotalPrice", "ItemsInCart", "RevenuePerItem",
          "HasCoupon", "OrderMonth", "PriceCategory"]].head(10))


# ============================================================
# STEP 5: ENCODING CATEGORICAL COLUMNS
# ============================================================
print("\n=== ENCODING ===")

# One-Hot Encoding for nominal categories (no order between them)
# Drop first to avoid multicollinearity (dummy variable trap)
ohe_cols = ["Product", "PaymentMethod", "ReferralSource", "CouponCode"]
df_encoded = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

print(f"Shape after OHE: {df_encoded.shape}")


# ============================================================
# STEP 6: COLLINEARITY CHECK
# ============================================================
print("\n=== COLLINEARITY CHECK ===")

num_df = df[["Quantity", "UnitPrice", "ItemsInCart",
             "TotalPrice", "RevenuePerItem", "HasCoupon", "OrderMonth"]]

corr_matrix = num_df.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr = [(col, row, upper_triangle.loc[row, col])
             for col in upper_triangle.columns
             for row in upper_triangle.index
             if upper_triangle.loc[row, col] > 0.80]

if high_corr:
    print("High correlation pairs (>0.80):")
    for a, b, val in high_corr:
        print(f"  {a} ↔ {b}: {val:.2f}")
else:
    print("No high collinearity found (all pairs < 0.80). Dataset is clean.")


# ============================================================
# STEP 7: PANDERA SCHEMA VALIDATION (Runtime Contract)
# ============================================================
print("\n=== SCHEMA VALIDATION ===")

schema = DataFrameSchema({
    "Quantity":       Column(int,   Check.in_range(1, 5)),
    "UnitPrice":      Column(float, Check.greater_than(0)),
    "ItemsInCart":    Column(int,   Check.in_range(1, 10)),
    "TotalPrice":     Column(float, Check.greater_than(0)),
    "HasCoupon":      Column(int,   Check.isin([0, 1])),
    "OrderMonth":     Column(np.int32, Check.in_range(1, 12)),
    "RevenuePerItem": Column(float, Check.greater_than(0)),
})

try:
    schema.validate(df, lazy=True)
    print("All schema checks passed.")
except pa.errors.SchemaErrors as e:
    print("Schema violations found:")
    print(e.failure_cases)


# ============================================================
# STEP 8: SAVE CLEANED DATASET
# ============================================================
df.to_csv("output/cleaned_dataset.csv", index=False)
df_encoded.to_csv("output/encoded_dataset.csv", index=False)
print("\nFiles saved to output/")
print("  → cleaned_dataset.csv  (with engineered features)")
print("  → encoded_dataset.csv  (ready for ML)")

print("\n=== DONE ===")