import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

MASTER_DATASET_PATH = "data/processed/master_dataset.csv"

IF_DATASET_PATH = "data/processed/financial_anomalies.csv"

LOF_DATASET_PATH = "data/processed/lof_anomalies.csv"

OUTPUT_PATH = "data/processed/ensemble_anomalies.csv"


# ============================================================
# CONFIGURATION
# ============================================================

# Isolation Forest receives slightly higher weight because it
# detects global anomalies.
IF_WEIGHT = 0.60

# LOF detects local anomalies.
LOF_WEIGHT = 0.40

# Risk percentile thresholds.
#
# These are ranking thresholds, NOT probabilities.

HIGH_THRESHOLD = 0.95

# Strong model agreement threshold.
BOTH_MODEL_HIGH_THRESHOLD = 0.90


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("ML ENSEMBLE ANOMALY DETECTION")
print("=" * 60)


master_df = pd.read_csv(
    MASTER_DATASET_PATH
)

if_df = pd.read_csv(
    IF_DATASET_PATH
)

lof_df = pd.read_csv(
    LOF_DATASET_PATH
)


print(
    "\nMaster projects:",
    len(master_df)
)

print(
    "Isolation Forest rows:",
    len(if_df)
)

print(
    "LOF rows:",
    len(lof_df)
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_master_columns = [
    "work_id"
]

required_if_columns = [
    "work_id",
    "anomaly_prediction",
    "is_anomaly",
    "anomaly_score"
]

required_lof_columns = [
    "work_id",
    "lof_prediction",
    "lof_is_anomaly",
    "lof_score"
]


def validate_columns(
    dataframe,
    required_columns,
    dataset_name
):

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            f"{dataset_name} is missing required columns: "
            + ", ".join(missing)
        )


validate_columns(
    master_df,
    required_master_columns,
    "Master dataset"
)

validate_columns(
    if_df,
    required_if_columns,
    "Isolation Forest dataset"
)

validate_columns(
    lof_df,
    required_lof_columns,
    "LOF dataset"
)


# ============================================================
# VALIDATE WORK IDS
# ============================================================

if not master_df["work_id"].is_unique:

    raise ValueError(
        "Master dataset contains duplicate work_id values."
    )


if not if_df["work_id"].is_unique:

    raise ValueError(
        "Isolation Forest output contains duplicate work_id values."
    )


if not lof_df["work_id"].is_unique:

    raise ValueError(
        "LOF output contains duplicate work_id values."
    )


# ============================================================
# SELECT ISOLATION FOREST DATA
# ============================================================

if_columns = [
    "work_id",
    "anomaly_prediction",
    "is_anomaly",
    "anomaly_score"
]


# Prefer the precomputed IF risk score if available.

if "if_risk_score" in if_df.columns:

    if_columns.append(
        "if_risk_score"
    )

elif "if_normalized_score" in if_df.columns:

    if_columns.append(
        "if_normalized_score"
    )


if_data = if_df[
    if_columns
].copy()


# ============================================================
# SELECT LOF DATA
# ============================================================

lof_columns = [
    "work_id",
    "lof_prediction",
    "lof_is_anomaly",
    "lof_score"
]


# Prefer the new rank-based LOF risk score.

if "lof_risk_score" in lof_df.columns:

    lof_columns.append(
        "lof_risk_score"
    )

elif "lof_normalized_score" in lof_df.columns:

    lof_columns.append(
        "lof_normalized_score"
    )


if "lof_pattern_frequency" in lof_df.columns:

    lof_columns.append(
        "lof_pattern_frequency"
    )


lof_data = lof_df[
    lof_columns
].copy()


# ============================================================
# START FROM MASTER DATASET
# ============================================================

df = master_df[
    ["work_id"]
].copy()


# ============================================================
# MERGE ISOLATION FOREST
# ============================================================

df = df.merge(
    if_data,
    on="work_id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# MERGE LOF
# ============================================================

df = df.merge(
    lof_data,
    on="work_id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# COVERAGE VALIDATION
# ============================================================

print(
    "\nMerged rows:",
    len(df)
)

print(
    "Unique work IDs:",
    df["work_id"].nunique()
)


if len(df) != len(master_df):

    raise ValueError(
        "Ensemble output does not contain all master projects."
    )


if df["work_id"].nunique() != len(master_df):

    raise ValueError(
        "Duplicate work IDs detected after ensemble merge."
    )


# ============================================================
# ROBUST BOOLEAN CONVERSION
# ============================================================

def to_binary_flag(series):

    # Handle numeric values.

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    result = pd.Series(
        0,
        index=series.index,
        dtype=int
    )

    numeric_mask = numeric.notna()

    result.loc[numeric_mask] = (
        numeric.loc[numeric_mask] != 0
    ).astype(int)

    # Handle strings such as:
    #
    # "True"
    # "False"
    # "yes"
    # "no"
    # "1"
    # "0"

    string_values = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    true_values = {
        "true",
        "yes",
        "y",
        "1",
        "anomaly",
        "anomalous"
    }

    string_mask = (
        ~numeric_mask
        &
        string_values.isin(true_values)
    )

    result.loc[string_mask] = 1

    return result


df["is_anomaly"] = to_binary_flag(
    df["is_anomaly"]
)

df["lof_is_anomaly"] = to_binary_flag(
    df["lof_is_anomaly"]
)


# ============================================================
# MODEL AGREEMENT
# ============================================================

df["both_models_anomaly"] = (

    (
        df["is_anomaly"] == 1
    )

    &

    (
        df["lof_is_anomaly"] == 1
    )

).astype(int)


df["any_model_anomaly"] = (

    (
        df["is_anomaly"] == 1
    )

    |

    (
        df["lof_is_anomaly"] == 1
    )

).astype(int)


# ============================================================
# CREATE IF RISK SCORE
# ============================================================

if "if_risk_score" in df.columns:

    df["if_risk_score"] = pd.to_numeric(
        df["if_risk_score"],
        errors="coerce"
    )

elif "if_normalized_score" in df.columns:

    df["if_risk_score"] = pd.to_numeric(
        df["if_normalized_score"],
        errors="coerce"
    )

else:

    # Isolation Forest anomaly_score:
    # lower = more anomalous in the existing pipeline.
    #
    # Convert to percentile risk.

    if_score = pd.to_numeric(
        df["anomaly_score"],
        errors="coerce"
    )

    if_rank = if_score.rank(
        method="average",
        ascending=True
    )

    df["if_risk_score"] = (
        1.0 -
        (
            if_rank /
            len(if_rank)
        )
    )


# ============================================================
# CREATE LOF RISK SCORE
# ============================================================

if "lof_risk_score" in df.columns:

    df["lof_risk_score"] = pd.to_numeric(
        df["lof_risk_score"],
        errors="coerce"
    )

elif "lof_normalized_score" in df.columns:

    df["lof_risk_score"] = pd.to_numeric(
        df["lof_normalized_score"],
        errors="coerce"
    )

else:

    # LOF:
    # lower score = more anomalous.
    #
    # Convert to percentile risk.

    lof_score = pd.to_numeric(
        df["lof_score"],
        errors="coerce"
    )

    lof_rank = lof_score.rank(
        method="average",
        ascending=True
    )

    df["lof_risk_score"] = (
        1.0 -
        (
            lof_rank /
            len(lof_rank)
        )
    )


# ============================================================
# HANDLE MISSING RISK SCORES
# ============================================================

df["if_risk_score"] = (
    df["if_risk_score"]
    .fillna(0)
    .clip(0.0, 1.0)
)


df["lof_risk_score"] = (
    df["lof_risk_score"]
    .fillna(0)
    .clip(0.0, 1.0)
)


# ============================================================
# CONVERT TO 0-100
# ============================================================

df["if_risk_score_100"] = (
    df["if_risk_score"] * 100
)

df["lof_risk_score_100"] = (
    df["lof_risk_score"] * 100
)


# ============================================================
# COMBINED ML RISK SCORE
# ============================================================

df["combined_anomaly_score"] = (

    IF_WEIGHT *
    df["if_risk_score"]

    +

    LOF_WEIGHT *
    df["lof_risk_score"]

)


# ============================================================
# COMBINED SCORE 0-100
# ============================================================

df["combined_risk_score"] = (

    df["combined_anomaly_score"] *
    100

)


# ============================================================
# MODEL AGREEMENT BONUS
# ============================================================

# Both models agreeing is stronger evidence than
# one model alone.

df["agreement_bonus"] = np.where(

    df["both_models_anomaly"] == 1,

    0.10,

    0.0

)


# Apply bonus but cap at 1.0.

df["ml_final_risk_score"] = (

    df["combined_anomaly_score"]
    +
    df["agreement_bonus"]

).clip(
    0.0,
    1.0
)


df["ml_final_risk_score_100"] = (

    df["ml_final_risk_score"] *
    100

)


# ============================================================
# ENSEMBLE ANOMALY FLAG
# ============================================================

# IMPORTANT:
#
# We do NOT use:
#
# combined_score >= 0.50
#
# because the risk score is percentile/rank based.
#
# Instead:
#
# 1. Both models anomalous = anomaly
# 2. IF anomalous = anomaly
# 3. LOF anomalous = anomaly
# 4. Very high combined risk = anomaly


df["ensemble_is_anomaly"] = np.where(

    (

        df["both_models_anomaly"] == 1

    )

    |

    (

        df["is_anomaly"] == 1

    )

    |

    (

        df["lof_is_anomaly"] == 1

    )

    |

    (

        df["ml_final_risk_score"]
        >= HIGH_THRESHOLD

    ),

    1,

    0

)


# ============================================================
# MODEL AGREEMENT CATEGORY
# ============================================================

def get_agreement(row):

    if (
        row["is_anomaly"] == 1
        and
        row["lof_is_anomaly"] == 1
    ):

        return "BOTH_MODELS"

    elif row["is_anomaly"] == 1:

        return "ISOLATION_FOREST_ONLY"

    elif row["lof_is_anomaly"] == 1:

        return "LOF_ONLY"

    else:

        return "NO_MODEL_ANOMALY"


df["model_agreement"] = df.apply(
    get_agreement,
    axis=1
)


# ============================================================
# ENSEMBLE RISK LEVEL
# ============================================================

# ============================================================
# FINAL ENSEMBLE RISK LEVEL
# ============================================================

def get_ensemble_level(row):

    score = row["ml_final_risk_score"]

    if_anomaly = row["is_anomaly"]

    lof_anomaly = row["lof_is_anomaly"]

    both = row["both_models_anomaly"]


    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------
    # Both independent anomaly models agree AND the combined
    # risk ranking is extremely high.
    #
    # This is the strongest ML evidence.
    #
    if (
        both == 1
        and
        score >= BOTH_MODEL_HIGH_THRESHOLD
    ):

        return "CRITICAL"


    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------
    # A single model identifies an anomaly AND the project
    # is very high in the combined ML ranking.
    #
    elif (
        (
            if_anomaly == 1
            or
            lof_anomaly == 1
        )
        and
        score >= HIGH_THRESHOLD
    ):

        return "HIGH"


    # --------------------------------------------------------
    # MEDIUM
    # --------------------------------------------------------
    # A model actually detected an anomaly, but the combined
    # ranking is not in the extreme tail.
    #
    elif (
        if_anomaly == 1
        or
        lof_anomaly == 1
    ):

        return "MEDIUM"


    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------
    # No anomaly model has classified this project as anomalous.
    #
    # A high ranking score alone does NOT make it HIGH/CRITICAL.
    #
    else:

        return "LOW"


df["ensemble_risk_level"] = df.apply(
    get_ensemble_level,
    axis=1
)


# ============================================================
# PRIMARY MODEL SOURCE
# ============================================================

def get_primary_source(row):

    if row["both_models_anomaly"] == 1:

        return "ISOLATION_FOREST + LOF"

    elif row["is_anomaly"] == 1:

        return "ISOLATION_FOREST"

    elif row["lof_is_anomaly"] == 1:

        return "LOF"

    elif (
        row["if_risk_score"]
        >=
        row["lof_risk_score"]
    ):

        return "ISOLATION_FOREST_RANK"

    else:

        return "LOF_RANK"


df["primary_ml_source"] = df.apply(
    get_primary_source,
    axis=1
)


# ============================================================
# ANOMALY RANK
# ============================================================

df["ensemble_anomaly_rank"] = (

    df["ml_final_risk_score"]
    .rank(
        method="min",
        ascending=False
    )
    .astype(int)

)


# ============================================================
# SORT RESULTS
# ============================================================

df = df.sort_values(
    by="ml_final_risk_score",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("ENSEMBLE ANOMALY DETECTION COMPLETE")
print("=" * 60)


print(
    "\nTotal Projects:",
    len(df)
)


print(
    "\nIsolation Forest Anomalies:",
    int(
        df["is_anomaly"].sum()
    )
)


print(
    "LOF Anomalies:",
    int(
        df["lof_is_anomaly"].sum()
    )
)


print(
    "Both Models Detected:",
    int(
        df["both_models_anomaly"].sum()
    )
)


print(
    "Any Model Detected:",
    int(
        df["any_model_anomaly"].sum()
    )
)


print(
    "Final Ensemble Anomalies:",
    int(
        df["ensemble_is_anomaly"].sum()
    )
)


# ============================================================
# RISK SCORE VALIDATION
# ============================================================

print(
    "\nML FINAL RISK SCORE"
)

print(
    "Min:",
    round(
        df["ml_final_risk_score"].min(),
        4
    )
)

print(
    "Max:",
    round(
        df["ml_final_risk_score"].max(),
        4
    )
)

print(
    "Mean:",
    round(
        df["ml_final_risk_score"].mean(),
        4
    )
)


# ============================================================
# TOP PROJECTS
# ============================================================

print(
    "\nTop Suspicious Projects:\n"
)


top_columns = [

    "work_id",

    "is_anomaly",

    "lof_is_anomaly",

    "both_models_anomaly",

    "model_agreement",

    "if_risk_score_100",

    "lof_risk_score_100",

    "combined_risk_score",

    "ml_final_risk_score_100",

    "ensemble_is_anomaly",

    "ensemble_risk_level",

    "primary_ml_source"

]


print(

    df[
        top_columns
    ].head(20)

)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

print(
    "\nENSEMBLE RISK LEVEL DISTRIBUTION\n"
)


print(
    df[
        "ensemble_risk_level"
    ].value_counts()
)


# ============================================================
# MODEL AGREEMENT
# ============================================================

print(
    "\nMODEL AGREEMENT ANALYSIS\n"
)


print(

    pd.crosstab(

        df["is_anomaly"],

        df["lof_is_anomaly"],

        rownames=[
            "Isolation Forest"
        ],

        colnames=[
            "LOF"
        ]

    )

)


print(
    "\nAgreement categories:\n"
)


print(
    df[
        "model_agreement"
    ].value_counts()
)


# ============================================================
# COVERAGE VALIDATION
# ============================================================

print(
    "\nCOVERAGE VALIDATION"
)


print(
    "Master rows:",
    len(master_df)
)


print(
    "IF rows:",
    len(if_df)
)


print(
    "LOF rows:",
    len(lof_df)
)


print(
    "Ensemble rows:",
    len(df)
)


print(
    "Unique ensemble IDs:",
    df["work_id"].nunique()
)


# ============================================================
# FINAL VALIDATION
# ============================================================

if len(df) != len(master_df):

    raise ValueError(
        "FINAL VALIDATION FAILED: "
        "row count mismatch."
    )


if df["work_id"].nunique() != len(master_df):

    raise ValueError(
        "FINAL VALIDATION FAILED: "
        "duplicate work IDs."
    )


if (
    df["ml_final_risk_score"].min()
    < 0
    or
    df["ml_final_risk_score"].max()
    > 1
):

    raise ValueError(
        "FINAL VALIDATION FAILED: "
        "risk score outside 0-1 range."
    )


print(
    "\n✓ PASS: All master projects preserved"
)

print(
    "✓ PASS: All work IDs are unique"
)

print(
    "✓ PASS: ML risk scores are within 0-1"
)

print(
    "✓ PASS: Ensemble validation completed"
)


print(
    "\nResults saved:",
    OUTPUT_PATH
)


print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)