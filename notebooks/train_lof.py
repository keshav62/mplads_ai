import pandas as pd
import numpy as np
import joblib

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = "data/processed/master_dataset.csv"

OUTPUT_FILE = "data/processed/lof_anomalies.csv"

MODEL_FILE = "models/local_outlier_factor.pkl"

SCALER_FILE = "models/lof_scaler.pkl"

IMPUTER_FILE = "models/lof_imputer.pkl"


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("LOF FINANCIAL ANOMALY DETECTION")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nMaster dataset loaded")
print("Total projects:", len(df))


# ============================================================
# VALIDATE WORK ID
# ============================================================

if "work_id" not in df.columns:
    raise ValueError(
        "work_id column not found in master_dataset.csv"
    )

if df["work_id"].duplicated().any():

    duplicate_count = df["work_id"].duplicated().sum()

    raise ValueError(
        f"Duplicate work_id values found: {duplicate_count}"
    )


# ============================================================
# FINANCIAL FEATURES
# ============================================================

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


# ============================================================
# VERIFY FEATURES
# ============================================================

missing_features = [
    col
    for col in financial_features
    if col not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing required features: "
        + ", ".join(missing_features)
    )


print("\nFinancial features:")

for feature in financial_features:
    print("  -", feature)


# ============================================================
# CREATE FEATURE MATRIX
# ============================================================

X_financial = df[
    financial_features
].copy()


# ============================================================
# CONVERT TO NUMERIC
# ============================================================

for column in financial_features:

    X_financial[column] = pd.to_numeric(
        X_financial[column],
        errors="coerce"
    )


# ============================================================
# REPLACE INFINITE VALUES
# ============================================================

X_financial = X_financial.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# MISSINGNESS INDICATORS
# ============================================================

missingness_features = [

    "sanction_amount",
    "total_expenditure",
    "expenditure_ratio",
    "payment_count",
    "unique_vendors",
    "payments_per_vendor"

]


print("\nCreating missingness indicators:")

for column in missingness_features:

    indicator_name = "missing_" + column

    X_financial[indicator_name] = (
        X_financial[column]
        .isna()
        .astype(int)
    )

    print(
        "  -",
        indicator_name
    )


# ============================================================
# MISSING VALUE REPORT
# ============================================================

print("\nMissing values before imputation:")

for column in financial_features:

    print(
        f"  {column}: "
        f"{X_financial[column].isna().sum()}"
    )


# ============================================================
# IMPUTATION
# ============================================================

print("\nApplying median imputation...")

imputer = SimpleImputer(
    strategy="median"
)

X_base = X_financial[
    financial_features
].copy()

X_imputed_base = imputer.fit_transform(
    X_base
)

X_imputed_base = pd.DataFrame(
    X_imputed_base,
    columns=financial_features,
    index=df.index
)


# ============================================================
# ADD MISSINGNESS INDICATORS
# ============================================================

indicator_columns = [
    "missing_" + column
    for column in missingness_features
]

X_indicators = X_financial[
    indicator_columns
].copy()


X_model = pd.concat(
    [
        X_imputed_base,
        X_indicators
    ],
    axis=1
)


print(
    "\nFinal feature count:",
    X_model.shape[1]
)


# ============================================================
# SCALE FEATURES
# ============================================================

print("\nScaling features...")

scaler = RobustScaler()

X_scaled_array = scaler.fit_transform(
    X_model
)


# IMPORTANT:
# Convert scaled NumPy array back into a DataFrame
# so sklearn always receives feature names.

X_scaled_df = pd.DataFrame(
    X_scaled_array,
    columns=X_model.columns,
    index=df.index
)


# ============================================================
# DETECT DUPLICATE FEATURE PATTERNS
# ============================================================

duplicate_rows = (
    X_scaled_df.duplicated().sum()
)

unique_rows = (
    len(
        X_scaled_df.drop_duplicates()
    )
)


print("\nFeature pattern analysis:")

print(
    "Total projects:",
    len(X_scaled_df)
)

print(
    "Duplicate rows:",
    duplicate_rows
)

print(
    "Unique feature patterns:",
    unique_rows
)


if unique_rows < 100:

    raise ValueError(
        "Too few unique feature patterns for LOF."
    )


# ============================================================
# CREATE UNIQUE FEATURE MATRIX
# ============================================================

unique_X = (
    X_scaled_df
    .drop_duplicates()
    .reset_index(drop=True)
)


print(
    "\nRunning LOF on",
    len(unique_X),
    "unique feature patterns."
)


# ============================================================
# SELECT NEIGHBOR COUNT
# ============================================================

n_neighbors = min(
    50,
    len(unique_X) - 1
)


print(
    "LOF n_neighbors:",
    n_neighbors
)


# ============================================================
# TRAIN LOF
# ============================================================

print("\nTraining LOF model...")


lof = LocalOutlierFactor(

    n_neighbors=n_neighbors,

    contamination=0.05,

    novelty=True

)


# Fit using DataFrame WITH feature names
lof.fit(unique_X.to_numpy())


# ============================================================
# SCORE ALL PROJECTS
# ============================================================

# score_samples:
#
# Higher value = more normal
# Lower value = more anomalous

lof_score_all = lof.score_samples(X_scaled_df.to_numpy())

# ============================================================
# PREDICT ALL PROJECTS
# ============================================================
lof_prediction_all = lof.predict(X_scaled_df.to_numpy())



# ============================================================
# CREATE ANOMALY FLAG
# ============================================================

df["lof_prediction"] = (
    lof_prediction_all
)

df["lof_is_anomaly"] = np.where(
    lof_prediction_all == -1,
    1,
    0
)


# ============================================================
# SAVE RAW SCORE
# ============================================================

df["lof_score"] = (
    lof_score_all
)


# ============================================================
# CREATE STABLE RISK SCORE
# ============================================================

# Lower LOF score = more anomalous.
#
# Convert scores into percentile/rank.
#
# This prevents extreme raw LOF values from
# dominating downstream risk calculations.

lof_series = pd.Series(
    lof_score_all,
    index=df.index
)


lof_rank = (
    lof_series
    .rank(
        method="average",
        ascending=True
    )
)


lof_percentile = (
    lof_rank /
    len(lof_rank)
)


# Reverse:
#
# lowest score  -> highest risk
# highest score -> lowest risk

df["lof_risk_score"] = (
    1.0 -
    lof_percentile
)


df["lof_risk_score"] = (
    df["lof_risk_score"]
    .clip(0.0, 1.0)
)


# ============================================================
# 0-100 VERSION
# ============================================================

df["lof_risk_score_100"] = (
    df["lof_risk_score"] * 100
)


# ============================================================
# ANOMALY RANK
# ============================================================

df["lof_anomaly_rank"] = (

    df["lof_risk_score"]
    .rank(
        method="min",
        ascending=False
    )
    .astype(int)

)


# ============================================================
# DUPLICATE PATTERN FREQUENCY
# ============================================================

pattern_counts = (
    X_scaled_df
    .groupby(
        list(X_scaled_df.columns),
        dropna=False
    )
    .size()
)


# Create stable pattern keys

pattern_keys = [
    tuple(row)
    for row in X_scaled_df.to_numpy()
]


pattern_frequency = pd.Series(
    pattern_keys
).map(
    pattern_counts
)


df["lof_pattern_frequency"] = (
    pattern_frequency.values
)


# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(
    lof,
    MODEL_FILE
)

joblib.dump(
    scaler,
    SCALER_FILE
)

joblib.dump(
    imputer,
    IMPUTER_FILE
)


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

total_projects = len(df)

total_anomalies = (
    df["lof_is_anomaly"] == 1
).sum()

normal_projects = (
    total_projects -
    total_anomalies
)


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("LOF MODEL TRAINING COMPLETE")
print("=" * 60)


print(
    "\nTotal projects:",
    total_projects
)

print(
    "Unique feature patterns:",
    unique_rows
)

print(
    "Duplicate feature rows:",
    duplicate_rows
)

print(
    "LOF anomalies detected:",
    total_anomalies
)

print(
    "Normal projects:",
    normal_projects
)

print(
    "Project anomaly percentage:",
    round(
        total_anomalies /
        total_projects *
        100,
        2
    ),
    "%"
)


# ============================================================
# SCORE VALIDATION
# ============================================================

print("\nLOF SCORE VALIDATION")


print(
    "Raw LOF min:",
    df["lof_score"].min()
)

print(
    "Raw LOF max:",
    df["lof_score"].max()
)

print(
    "Risk score min:",
    df["lof_risk_score"].min()
)

print(
    "Risk score max:",
    df["lof_risk_score"].max()
)

print(
    "Risk score mean:",
    round(
        df["lof_risk_score"].mean(),
        4
    )
)


# ============================================================
# TOP PROJECTS
# ============================================================

print("\nTop LOF Risk Projects:\n")


print(

    df.sort_values(
        "lof_risk_score",
        ascending=False
    )[

        [

            "work_id",
            "lof_score",
            "lof_risk_score",
            "lof_risk_score_100",
            "lof_is_anomaly",
            "lof_anomaly_rank",
            "lof_pattern_frequency"

        ]

    ].head(20)

)


# ============================================================
# SAVE INFORMATION
# ============================================================

print(
    "\nModel saved:",
    MODEL_FILE
)

print(
    "Scaler saved:",
    SCALER_FILE
)

print(
    "Imputer saved:",
    IMPUTER_FILE
)

print(
    "Results saved:",
    OUTPUT_FILE
)


# ============================================================
# ROW COUNT VALIDATION
# ============================================================

print("\nRow-count validation:")


if total_projects == 13694:

    print(
        "✓ PASS: All 13,694 projects processed"
    )

else:

    print(
        "⚠ WARNING: Expected 13,694 projects "
        f"but got {total_projects}"
    )


# ============================================================
# UNIQUE ID VALIDATION
# ============================================================

if (
    df["work_id"].nunique()
    == total_projects
):

    print(
        "✓ PASS: All work IDs are unique"
    )

else:

    print(
        "⚠ WARNING: Duplicate work IDs detected"
    )


print("\n" + "=" * 60)