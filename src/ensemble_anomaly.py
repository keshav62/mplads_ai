import pandas as pd
import numpy as np


# ==========================================
# LOAD ISOLATION FOREST RESULTS
# ==========================================

if_df = pd.read_csv(
    "data/processed/financial_anomalies.csv"
)


# ==========================================
# LOAD LOF RESULTS
# ==========================================

lof_df = pd.read_csv(
    "data/processed/lof_anomalies.csv"
)


# ==========================================
# SELECT REQUIRED COLUMNS
# ==========================================

if_columns = [

    "work_id",

    "anomaly_prediction",

    "is_anomaly",

    "anomaly_score"

]


lof_columns = [

    "work_id",

    "lof_prediction",

    "lof_is_anomaly",

    "lof_score"

]


if_data = if_df[
    if_columns
].copy()


lof_data = lof_df[
    lof_columns
].copy()


# ==========================================
# MERGE BOTH MODEL RESULTS
# ==========================================

df = if_data.merge(

    lof_data,

    on="work_id",

    how="inner"

)


# ==========================================
# CREATE COMBINED ANOMALY FLAGS
# ==========================================


# Both models detected anomaly

df["both_models_anomaly"] = np.where(

    (df["is_anomaly"] == 1) &

    (df["lof_is_anomaly"] == 1),

    1,

    0

)


# Either model detected anomaly

df["any_model_anomaly"] = np.where(

    (df["is_anomaly"] == 1) |

    (df["lof_is_anomaly"] == 1),

    1,

    0

)


# ==========================================
# CREATE ENSEMBLE ANOMALY SCORE
# ==========================================

# Normalize Isolation Forest score

if_min = df["anomaly_score"].min()

if_max = df["anomaly_score"].max()


df["if_normalized_score"] = (

    df["anomaly_score"] - if_min

) / (

    if_max - if_min

)


# Normalize LOF score

lof_min = df["lof_score"].min()

lof_max = df["lof_score"].max()


df["lof_normalized_score"] = (

    df["lof_score"] - lof_min

) / (

    lof_max - lof_min

)


# ==========================================
# INVERT SCORES
# ==========================================

# Lower score = more suspicious
# So invert them

df["if_risk_score"] = (

    1 - df["if_normalized_score"]

)


df["lof_risk_score"] = (

    1 - df["lof_normalized_score"]

)


# ==========================================
# COMBINED AI SCORE
# ==========================================

df["combined_anomaly_score"] = (

    0.6 * df["if_risk_score"]

    +

    0.4 * df["lof_risk_score"]

)


# ==========================================
# FINAL ENSEMBLE PREDICTION
# ==========================================

# ==========================================
# ENSEMBLE ANOMALY CLASSIFICATION
# ==========================================

df["ensemble_is_anomaly"] = np.where(

    (
        (df["both_models_anomaly"] == 1)
        |
        (df["combined_anomaly_score"] >= 0.50)
    ),

    1,

    0

)

# ==========================================
# ENSEMBLE RISK LEVEL
# ==========================================

def get_ensemble_level(row):

    if row["both_models_anomaly"] == 1:
        return "CRITICAL"

    elif row["combined_anomaly_score"] >= 0.50:
        return "HIGH"

    elif row["combined_anomaly_score"] >= 0.35:
        return "MEDIUM"

    else:
        return "LOW"


df["ensemble_risk_level"] = df.apply(

    get_ensemble_level,

    axis=1

)


# ==========================================
# SAVE RESULTS
# ==========================================

df.to_csv(

    "data/processed/ensemble_anomalies.csv",

    index=False

)


# ==========================================
# RESULTS
# ==========================================

print("\n" + "=" * 60)

print("ENSEMBLE ANOMALY DETECTION COMPLETE")

print("=" * 60)


print(

    "\nTotal Projects:",

    len(df)

)


print(

    "\nIsolation Forest Anomalies:",

    df["is_anomaly"].sum()

)


print(

    "LOF Anomalies:",

    df["lof_is_anomaly"].sum()

)


print(

    "Both Models Detected:",

    df["both_models_anomaly"].sum()

)


print(

    "Final Ensemble Anomalies:",

    df["ensemble_is_anomaly"].sum()

)


print("\nTop Suspicious Projects:\n")


print(

    df.sort_values(

        by="combined_anomaly_score",

        ascending=False

    )[

        [

            "work_id",

            "is_anomaly",

            "lof_is_anomaly",

            "combined_anomaly_score"

        ]

    ].head(20)

)


print(

    "\nResults saved:",

    "data/processed/ensemble_anomalies.csv"

)

print("Isolation Forest rows:", len(if_df))

print("LOF rows:", len(lof_df))

print("Merged rows:", len(df))

print("Unique work IDs:", df["work_id"].nunique())

print("\nENSEMBLE RISK LEVEL DISTRIBUTION\n")

print(
    df["ensemble_risk_level"].value_counts()
)

print("\nMODEL AGREEMENT ANALYSIS\n")

print(

    pd.crosstab(

        df["is_anomaly"],

        df["lof_is_anomaly"],

        rownames=["Isolation Forest"],

        colnames=["LOF"]

    )

)