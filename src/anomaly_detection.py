import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest

import joblib
import os


# ==========================================
# LOAD MASTER DATASET
# ==========================================

master_df = pd.read_csv(
    "data/processed/master_dataset.csv"
)


# ==========================================
# SELECT FINANCIAL RECORDS
# ==========================================

financial_df = master_df[
    master_df["total_expenditure"].notna()
].copy()


# ==========================================
# SELECT FEATURES
# ==========================================

feature_columns = [

    "recommended_amount",

    "sanction_amount",

    "total_expenditure",

    "expenditure_ratio",

    "payment_count",

    "unique_vendors",

    "payments_per_vendor",

    "has_sanction"

]


X_financial = financial_df[
    feature_columns
].copy()


# ==========================================
# HANDLE MISSING VALUES
# ==========================================

X_financial = X_financial.replace(
    [np.inf, -np.inf],
    np.nan
)


X_financial["sanction_amount"] = (
    X_financial["sanction_amount"]
    .fillna(0)
)


X_financial["expenditure_ratio"] = (
    X_financial["expenditure_ratio"]
    .fillna(0)
)


X_financial = X_financial.fillna(0)


# ==========================================
# LOAD SCALER
# ==========================================

scaler = joblib.load(
    "models/robust_scaler.pkl"
)


# ==========================================
# SCALE DATA
# ==========================================

X_scaled = scaler.transform(
    X_financial
)


# ==========================================
# CREATE ISOLATION FOREST
# ==========================================

model = IsolationForest(

    n_estimators=300,

    contamination=0.05,

    random_state=42,

    n_jobs=-1

)


# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(X_scaled)


# ==========================================
# PREDICT ANOMALIES
# ==========================================

financial_df["anomaly_prediction"] = (
    model.predict(X_scaled)
)


# ==========================================
# CREATE ANOMALY LABEL
# ==========================================

financial_df["is_anomaly"] = (
    financial_df["anomaly_prediction"] == -1
).astype(int)


# ==========================================
# GET ANOMALY SCORE
# ==========================================

financial_df["anomaly_score"] = (
    model.decision_function(X_scaled)
)


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(

    model,

    "models/isolation_forest.pkl"

)


# ==========================================
# SAVE RESULTS
# ==========================================

financial_df.to_csv(

    "data/processed/financial_anomalies.csv",

    index=False

)


# ==========================================
# RESULTS
# ==========================================

print("\nMODEL TRAINING COMPLETE")

print(
    "\nTotal financial records:",
    len(financial_df)
)


print(
    "Anomalies detected:",
    financial_df["is_anomaly"].sum()
)


print(
    "Normal records:",
    (financial_df["is_anomaly"] == 0).sum()
)


print(
    "\nModel saved: models/isolation_forest.pkl"
)


print(
    "Results saved: data/processed/financial_anomalies.csv"
)


# ==========================================
# ANALYZE DETECTED ANOMALIES
# ==========================================

anomalies = financial_df[
    financial_df["is_anomaly"] == 1
].copy()


print("\n" + "=" * 60)

print("ANOMALY ANALYSIS")

print("=" * 60)


print("\nTotal anomalies:")

print(
    len(anomalies)
)


# Select important columns

analysis_columns = [

    "work_id",

    "state",

    "constituency",

    "recommended_amount",

    "sanction_amount",

    "total_expenditure",

    "expenditure_ratio",

    "payment_count",

    "unique_vendors",

    "payments_per_vendor",

    "anomaly_score"

]


print("\nTop suspicious records:\n")


print(
    anomalies[
        analysis_columns
    ]
    .sort_values(
        by="anomaly_score"
    )
    .head(10)
)


# ==========================================
# SAVE TOP ANOMALIES
# ==========================================

anomalies[
    analysis_columns
].sort_values(
    by="anomaly_score"
).to_csv(

    "data/processed/detected_anomalies.csv",

    index=False

)


print(
    "\nDetected anomalies saved successfully!"
)