import pandas as pd
import numpy as np

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

import joblib


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/processed/master_dataset.csv"
)
# Keep only projects that have financial expenditure data
df = df[
    df["total_expenditure"].notna()
].copy()

# ==========================================
# SELECT FINANCIAL FEATURES
# ==========================================

financial_features = [

    "recommended_amount",

    "sanction_amount",

    "total_expenditure",

    "expenditure_ratio",

    "payment_count",

    "unique_vendors",

    "payments_per_vendor",

    "has_sanction"

]


# Keep only required columns

X_financial = df[
    financial_features
].copy()


# ==========================================
# HANDLE MISSING VALUES
# ==========================================

X_financial = X_financial.replace(
    [np.inf, -np.inf],
    np.nan
)


X_financial = X_financial.fillna(
    X_financial.median()
)


# ==========================================
# SCALE FEATURES
# ==========================================

scaler = RobustScaler()

X_scaled = scaler.fit_transform(
    X_financial
)


# ==========================================
# TRAIN LOF MODEL
# ==========================================

lof = LocalOutlierFactor(

    n_neighbors=50,

    contamination=0.05
)


# LOF prediction

lof_prediction = lof.fit_predict(
    X_scaled
)


# ==========================================
# LOF SCORES
# ==========================================

# Convert negative LOF scores
# More negative = more suspicious

lof_score = lof.negative_outlier_factor_


# ==========================================
# CREATE RESULTS
# ==========================================

df["lof_prediction"] = lof_prediction


df["lof_is_anomaly"] = np.where(

    lof_prediction == -1,

    1,

    0

)


df["lof_score"] = lof_score


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(

    lof,

    "models/local_outlier_factor.pkl"

)


# ==========================================
# SAVE RESULTS
# ==========================================

df.to_csv(

    "data/processed/lof_anomalies.csv",

    index=False

)


# ==========================================
# RESULTS
# ==========================================

total_anomalies = (

    df["lof_is_anomaly"] == 1

).sum()


print("\nLOF MODEL TRAINING COMPLETE\n")


print(
    "Total records:",
    len(df)
)


print(
    "LOF anomalies detected:",
    total_anomalies
)


print(
    "Normal records:",
    len(df) - total_anomalies
)


print(
    "\nModel saved:",
    "models/local_outlier_factor.pkl"
)


print(
    "Results saved:",
    "data/processed/lof_anomalies.csv"
)