import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "data/processed/master_dataset.csv"

OUTPUT_PATH = "data/processed/financial_anomalies.csv"

MODEL_PATH = "models/isolation_forest.pkl"

SCALER_PATH = "models/isolation_forest_scaler.pkl"

IMPUTER_PATH = "models/isolation_forest_imputer.pkl"


FEATURE_COLUMNS = [
    "recommended_amount",
    "sanction_amount",
    "total_expenditure",
    "expenditure_ratio",
    "payment_count",
    "unique_vendors",
    "payments_per_vendor",
    "has_sanction",
]


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)

os.makedirs("data/processed", exist_ok=True)


# ============================================================
# LOAD MASTER DATASET
# ============================================================

print("\n" + "=" * 60)
print("ISOLATION FOREST FINANCIAL ANOMALY DETECTION")
print("=" * 60)

master_df = pd.read_csv(INPUT_PATH)

print("\nMaster dataset loaded")

print("Total projects:", len(master_df))


# ============================================================
# VALIDATE WORK IDS
# ============================================================

if "work_id" not in master_df.columns:

    raise ValueError(
        "work_id column is missing from master_dataset.csv"
    )


duplicate_ids = master_df["work_id"].duplicated().sum()

if duplicate_ids > 0:

    raise ValueError(
        f"Duplicate work_id values found: {duplicate_ids}"
    )


# ============================================================
# VALIDATE FEATURES
# ============================================================

missing_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in master_df.columns
]

if missing_features:

    raise ValueError(
        "Missing required features: "
        + ", ".join(missing_features)
    )


# ============================================================
# CREATE FEATURE DATAFRAME
# ============================================================

X = master_df[FEATURE_COLUMNS].copy()


# ============================================================
# CONVERT TO NUMERIC
# ============================================================

for column in FEATURE_COLUMNS:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# ============================================================
# REPLACE INFINITE VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# MISSINGNESS INDICATORS
# ============================================================

print("\nCreating missing-data indicators...")

missing_columns = [
    "sanction_amount",
    "total_expenditure",
    "expenditure_ratio",
    "payment_count",
    "unique_vendors",
    "payments_per_vendor",
]


for column in missing_columns:

    indicator_name = f"{column}_missing"

    X[indicator_name] = (
        X[column].isna().astype(int)
    )


# ============================================================
# DISPLAY MISSING VALUES
# ============================================================

print("\nMissing values before imputation:")

for column in FEATURE_COLUMNS:

    count = X[column].isna().sum()

    print(
        f"  {column}: {count}"
    )


# ============================================================
# IMPUTATION
# ============================================================

print("\nApplying median imputation...")

imputer = SimpleImputer(
    strategy="median"
)

X_imputed = imputer.fit_transform(X)


# ============================================================
# SCALING
# ============================================================

print("Applying RobustScaler...")

scaler = RobustScaler()

X_scaled = scaler.fit_transform(
    X_imputed
)


# ============================================================
# ISOLATION FOREST
# ============================================================

print("\nTraining Isolation Forest...")

model = IsolationForest(

    n_estimators=300,

    contamination=0.05,

    random_state=42,

    n_jobs=-1,
)


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(X_scaled)


# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(
    X_scaled
)


decision_scores = model.decision_function(
    X_scaled
)


# ============================================================
# CREATE OUTPUT
# ============================================================

result_df = master_df.copy()


result_df["anomaly_prediction"] = (
    predictions
)


result_df["is_anomaly"] = (
    predictions == -1
).astype(int)


result_df["anomaly_score"] = (
    decision_scores
)


# ============================================================
# NORMALIZED RISK SCORE
# ============================================================

score_min = decision_scores.min()

score_max = decision_scores.max()


if score_max > score_min:

    result_df["if_normalized_score"] = (

        (decision_scores - score_min)

        / (score_max - score_min)

    )

else:

    result_df["if_normalized_score"] = 0.0


# More anomalous = higher risk

result_df["if_risk_score"] = (

    1 - result_df["if_normalized_score"]

) * 100


result_df["if_risk_score"] = (

    result_df["if_risk_score"]

    .clip(0, 100)

    .round(2)

)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

joblib.dump(
    imputer,
    IMPUTER_PATH
)


# ============================================================
# SAVE RESULTS
# ============================================================

result_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

total_projects = len(result_df)

total_anomalies = (
    result_df["is_anomaly"] == 1
).sum()

normal_projects = (
    total_projects - total_anomalies
)


print("\n" + "=" * 60)
print("ISOLATION FOREST TRAINING COMPLETE")
print("=" * 60)

print(
    "\nTotal projects:",
    total_projects
)

print(
    "Isolation Forest anomalies:",
    total_anomalies
)

print(
    "Normal projects:",
    normal_projects
)

print(
    "Anomaly percentage:",
    round(
        total_anomalies
        / total_projects
        * 100,
        2
    ),
    "%"
)

print(
    "\nModel saved:",
    MODEL_PATH
)

print(
    "Scaler saved:",
    SCALER_PATH
)

print(
    "Imputer saved:",
    IMPUTER_PATH
)

print(
    "Results saved:",
    OUTPUT_PATH
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("DATA INTEGRITY VALIDATION")
print("=" * 60)

print(
    "Master rows:",
    len(master_df)
)

print(
    "Output rows:",
    len(result_df)
)

print(
    "Master unique work IDs:",
    master_df["work_id"].nunique()
)

print(
    "Output unique work IDs:",
    result_df["work_id"].nunique()
)


if (
    len(master_df) == len(result_df)
    and
    master_df["work_id"].nunique()
    == result_df["work_id"].nunique()
):

    print(
        "\n✓ PASS: All projects preserved"
    )

else:

    raise ValueError(
        "\n✗ FAIL: Project rows were lost"
    )