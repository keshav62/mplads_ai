import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MASTER_DATASET_PATH = "data/processed/master_dataset.csv"

ML_DATASET_PATH = "data/processed/ensemble_anomalies.csv"

RULE_DATASET_PATH = "data/processed/rule_based_risk_analysis.csv"

FINANCIAL_DATASET_PATH = "data/processed/financial_risk_analysis.csv"

STATISTICAL_DATASET_PATH = (
    "data/processed/statistical_financial_anomaly_analysis.csv"
)

OUTPUT_PATH = (
    "data/processed/final_unified_risk_analysis_v2.csv"
)


# ============================================================
# HELPER FUNCTION
# SAFE CSV LOADING
# ============================================================

def load_csv_if_exists(path, dataset_name):

    print(f"\nLoading {dataset_name}...")

    if not os.path.exists(path):

        print(f"WARNING: File not found: {path}")

        return None

    df = pd.read_csv(path)

    print(f"Total projects: {len(df)}")

    print(f"Columns available:")

    print(df.columns.tolist())

    return df


# ============================================================
# FIND FIRST AVAILABLE FILE
# ============================================================

def load_first_available(paths, dataset_name):

    print(f"\nLoading {dataset_name}...")

    for path in paths:

        if os.path.exists(path):

            print(f"Using file: {path}")

            df = pd.read_csv(path)

            print(f"Total projects: {len(df)}")

            return df

    print(f"WARNING: No valid file found for {dataset_name}")

    return None


# ============================================================
# SAFE NUMERIC CONVERSION
# ============================================================

def safe_numeric(df, column):

    if column not in df.columns:

        df[column] = 0

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    df[column] = df[column].replace(
        [np.inf, -np.inf],
        np.nan
    )

    df[column] = df[column].fillna(0)

    return df


# ============================================================
# NORMALIZE SCORE TO 0-100
# ============================================================

def normalize_score(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    max_value = series.max()

    if max_value <= 0:

        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    normalized = (
        series / max_value
    ) * 100

    return normalized.clip(0, 100)


# ============================================================
# RISK LEVEL TO NUMERIC SCORE
# FALLBACK WHEN SCORE COLUMN IS NOT AVAILABLE
# ============================================================

def risk_level_to_score(level):

    if pd.isna(level):

        return 0

    level = str(level).upper()

    mapping = {

        "LOW": 0,

        "MEDIUM": 40,

        "HIGH": 70,

        "CRITICAL": 100

    }

    return mapping.get(level, 0)


# ============================================================
# GET RISK LEVEL
# ============================================================

def get_final_risk_level(score):

    if score >= 75:

        return "CRITICAL"

    elif score >= 50:

        return "HIGH"

    elif score >= 25:

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================
# GET CONFIDENCE
# ============================================================

def get_risk_confidence(active_engines):

    if active_engines >= 4:

        return "VERY HIGH"

    elif active_engines == 3:

        return "HIGH"

    elif active_engines == 2:

        return "MEDIUM"

    elif active_engines == 1:

        return "LOW"

    else:

        return "VERY LOW"


# ============================================================
# CONSENSUS BONUS
# ============================================================

def get_consensus_bonus(active_engines):

    if active_engines >= 4:

        return 20

    elif active_engines == 3:

        return 12

    elif active_engines == 2:

        return 5

    else:

        return 0


# ============================================================
# PRIMARY RISK SOURCE
# ============================================================

def get_primary_risk_source(row):

    scores = {

        "ML Anomaly Engine":
            row["ml_normalized_score"],

        "Rule-Based Risk Engine":
            row["rule_normalized_score"],

        "Financial Risk Engine":
            row["financial_normalized_score"],

        "Statistical Anomaly Engine":
            row["statistical_normalized_score"]

    }

    max_source = max(
        scores,
        key=scores.get
    )

    max_score = scores[max_source]

    if max_score <= 0:

        return "No Significant Risk"

    return max_source


# ============================================================
# COMBINE RISK FACTORS
# ============================================================

def combine_risk_factors(row):

    factors = []

    factor_columns = [

        "risk_factors",

        "financial_risk_factors",

        "statistical_anomaly_factors"

    ]

    for column in factor_columns:

        if column not in row.index:

            continue

        value = row[column]

        if pd.isna(value):

            continue

        value = str(value)

        if value in ["[]", "", "nan", "None"]:

            continue

        factors.append(value)

    # Remove duplicates while preserving order

    unique_factors = []

    for factor in factors:

        if factor not in unique_factors:

            unique_factors.append(factor)

    return unique_factors


# ============================================================
# MAIN FUNCTION
# ============================================================

def build_unified_risk_engine():

    print("\n" + "=" * 60)

    print("UNIFIED RISK INTELLIGENCE ENGINE V2")

    print("=" * 60)


    # ========================================================
    # LOAD MASTER DATASET
    # ========================================================

    master_df = load_csv_if_exists(

        MASTER_DATASET_PATH,

        "master dataset"

    )

    if master_df is None:

        raise FileNotFoundError(

            f"Master dataset not found: {MASTER_DATASET_PATH}"

        )


    # ========================================================
    # CHECK WORK_ID
    # ========================================================

    if "work_id" not in master_df.columns:

        raise ValueError(

            "Master dataset must contain 'work_id'"

        )


    # ========================================================
    # LOAD ML DATASET
    # ========================================================

    ml_df = load_first_available(

        [

            "data/processed/ensemble_anomalies.csv",

            "data/processed/ensemble_anomaly_analysis.csv",

            "data/processed/detected_anomalies.csv"

        ],

        "ML anomaly dataset"

    )


    # ========================================================
    # LOAD RULE-BASED DATASET
    # ========================================================

    rule_df = load_first_available(

        [

            "data/processed/rule_based_risk_analysis.csv",

            "data/processed/risk_scored_projects.csv",

            "data/processed/final_ai_risk_analysis.csv"

        ],

        "rule-based risk dataset"

    )


    # ========================================================
    # LOAD FINANCIAL DATASET
    # ========================================================

    financial_df = load_first_available(

        [

            "data/processed/financial_risk_analysis.csv",

            "data/processed/financial_anomalies.csv"

        ],

        "financial risk dataset"

    )


    # ========================================================
    # LOAD STATISTICAL DATASET
    # ========================================================

    statistical_df = load_first_available(

        [

            "data/processed/statistical_financial_anomaly_analysis.csv",

            "data/processed/statistical_anomaly_analysis.csv"

        ],

        "statistical anomaly dataset"

    )


    # ========================================================
    # START WITH MASTER DATASET
    # ========================================================

    df = master_df.copy()


    print("\n" + "=" * 60)

    print("MERGING RISK ENGINES")

    print("=" * 60)


    # ========================================================
    # ML ENGINE
    # ========================================================

    if ml_df is not None and "work_id" in ml_df.columns:


        ml_score_column = None


        possible_ml_scores = [

            "combined_anomaly_score",

            "anomaly_score",

            "if_risk_score",

            "lof_risk_score"

        ]


        for column in possible_ml_scores:

            if column in ml_df.columns:

                ml_score_column = column

                break


        if ml_score_column is not None:

            print(

                f"\nML score column detected: "

                f"{ml_score_column}"

            )


            ml_merge = ml_df[

                ["work_id", ml_score_column]

            ].copy()


            ml_merge = ml_merge.rename(

                columns={

                    ml_score_column:
                        "ml_anomaly_score"

                }

            )


            df = df.merge(

                ml_merge,

                on="work_id",

                how="left"

            )


        else:

            print(

                "\nWARNING: No ML score column found"

            )

            df["ml_anomaly_score"] = 0


    else:

        print(

            "\nWARNING: ML dataset unavailable"

        )

        df["ml_anomaly_score"] = 0


    # ========================================================
    # RULE-BASED ENGINE
    # ========================================================

    if rule_df is not None and "work_id" in rule_df.columns:


        rule_score_column = None


        possible_rule_scores = [

            "rule_risk_score",

            "risk_score"

        ]


        for column in possible_rule_scores:

            if column in rule_df.columns:

                rule_score_column = column

                break


        if rule_score_column is not None:

            print(

                f"\nRule score column detected: "

                f"{rule_score_column}"

            )


            columns_to_keep = [

                "work_id",

                rule_score_column

            ]


            if "risk_factors" in rule_df.columns:

                columns_to_keep.append(
                    "risk_factors"
                )


            rule_merge = rule_df[
                columns_to_keep
            ].copy()


            rename_columns = {

                rule_score_column:
                    "rule_risk_score"

            }


            df = df.merge(

                rule_merge.rename(
                    columns=rename_columns
                ),

                on="work_id",

                how="left"

            )


        else:

            print(

                "\nWARNING: No rule-based score column found"

            )

            df["rule_risk_score"] = 0


    else:

        print(

            "\nWARNING: Rule-based dataset unavailable"

        )

        df["rule_risk_score"] = 0


    # ========================================================
    # FINANCIAL ENGINE
    # ========================================================

    if (

        financial_df is not None

        and

        "work_id" in financial_df.columns

    ):


        financial_columns = [

            "work_id",

            "financial_risk_score"

        ]


        if (

            "financial_risk_level"

            in financial_df.columns

        ):

            financial_columns.append(
                "financial_risk_level"
            )


        if (

            "financial_risk_factors"

            in financial_df.columns

        ):

            financial_columns.append(
                "financial_risk_factors"
            )


        financial_merge = financial_df[
            financial_columns
        ].copy()


        df = df.merge(

            financial_merge,

            on="work_id",

            how="left"

        )


    else:

        print(

            "\nWARNING: Financial dataset unavailable"

        )

        df["financial_risk_score"] = 0


    # ========================================================
    # STATISTICAL ENGINE
    # ========================================================

    if (

        statistical_df is not None

        and

        "work_id" in statistical_df.columns

    ):


        statistical_columns = [

            "work_id",

            "statistical_anomaly_score"

        ]


        if (

            "statistical_anomaly_level"

            in statistical_df.columns

        ):

            statistical_columns.append(
                "statistical_anomaly_level"
            )


        if (

            "statistical_anomaly_factors"

            in statistical_df.columns

        ):

            statistical_columns.append(
                "statistical_anomaly_factors"
            )


        statistical_merge = statistical_df[
            statistical_columns
        ].copy()


        df = df.merge(

            statistical_merge,

            on="work_id",

            how="left"

        )


    else:

        print(

            "\nWARNING: Statistical dataset unavailable"

        )

        df["statistical_anomaly_score"] = 0


    # ========================================================
    # ENSURE ALL SCORE COLUMNS EXIST
    # ========================================================

    score_columns = [

        "ml_anomaly_score",

        "rule_risk_score",

        "financial_risk_score",

        "statistical_anomaly_score"

    ]


    for column in score_columns:

        df = safe_numeric(
            df,
            column
        )


    # ========================================================
    # REMOVE DUPLICATE WORK IDS
    # ========================================================

    df = df.drop_duplicates(
        subset=["work_id"]
    )


    print(

        f"\nTotal projects after merge: "

        f"{len(df)}"

    )


    # ========================================================
    # NORMALIZE SCORES
    # ========================================================

    print("\n" + "=" * 60)

    print("NORMALIZING RISK SCORES")

    print("=" * 60)


    df["ml_normalized_score"] = normalize_score(

        df["ml_anomaly_score"]

    )


    df["rule_normalized_score"] = normalize_score(

        df["rule_risk_score"]

    )


    df["financial_normalized_score"] = normalize_score(

        df["financial_risk_score"]

    )


    df["statistical_normalized_score"] = normalize_score(

        df["statistical_anomaly_score"]

    )


    # ========================================================
    # RISK WEIGHTS
    # ========================================================

    ML_WEIGHT = 0.35

    RULE_WEIGHT = 0.25

    FINANCIAL_WEIGHT = 0.20

    STATISTICAL_WEIGHT = 0.20


    print("\nRISK WEIGHTS")

    print(
        f"ML Anomaly Score:         "
        f"{ML_WEIGHT * 100:.0f}%"
    )

    print(
        f"Rule-Based Risk Score:    "
        f"{RULE_WEIGHT * 100:.0f}%"
    )

    print(
        f"Financial Risk Score:     "
        f"{FINANCIAL_WEIGHT * 100:.0f}%"
    )

    print(
        f"Statistical Anomaly:      "
        f"{STATISTICAL_WEIGHT * 100:.0f}%"
    )


    # ========================================================
    # BASE RISK SCORE
    # ========================================================

    df["base_risk_score"] = (

        df["ml_normalized_score"]
        * ML_WEIGHT

        +

        df["rule_normalized_score"]
        * RULE_WEIGHT

        +

        df["financial_normalized_score"]
        * FINANCIAL_WEIGHT

        +

        df["statistical_normalized_score"]
        * STATISTICAL_WEIGHT

    )


    # ========================================================
    # ENGINE DETECTION FLAGS
    # ========================================================

    ML_THRESHOLD = 30

    RULE_THRESHOLD = 25

    FINANCIAL_THRESHOLD = 25

    STATISTICAL_THRESHOLD = 25


    df["ml_detected"] = (

        df["ml_normalized_score"]
        >= ML_THRESHOLD

    ).astype(int)


    df["rule_detected"] = (

        df["rule_normalized_score"]
        >= RULE_THRESHOLD

    ).astype(int)


    df["financial_detected"] = (

        df["financial_normalized_score"]
        >= FINANCIAL_THRESHOLD

    ).astype(int)


    df["statistical_detected"] = (

        df["statistical_normalized_score"]
        >= STATISTICAL_THRESHOLD

    ).astype(int)


    # ========================================================
    # ACTIVE RISK ENGINES
    # ========================================================

    df["active_risk_engines"] = (

        df["ml_detected"]

        +

        df["rule_detected"]

        +

        df["financial_detected"]

        +

        df["statistical_detected"]

    )


    # ========================================================
    # CONSENSUS BONUS
    # ========================================================

    df["consensus_bonus"] = (

        df["active_risk_engines"]

        .apply(
            get_consensus_bonus
        )

    )


    # ========================================================
    # FINAL AI RISK SCORE
    # ========================================================

    df["final_ai_risk_score"] = (

        df["base_risk_score"]

        +

        df["consensus_bonus"]

    ).clip(

        0,

        100

    ).round(2)


    # ========================================================
    # FINAL RISK LEVEL
    # ========================================================

    df["final_ai_risk_level"] = (

        df["final_ai_risk_score"]

        .apply(
            get_final_risk_level
        )

    )


    # ========================================================
    # RISK DETECTION CONFIDENCE
    # ========================================================

    df["risk_detection_confidence"] = (

        df["active_risk_engines"]

        .apply(
            get_risk_confidence
        )

    )


    # ========================================================
    # PRIMARY RISK SOURCE
    # ========================================================

    df["primary_risk_source"] = (

        df.apply(

            get_primary_risk_source,

            axis=1

        )

    )


    # ========================================================
    # COMBINED RISK FACTORS
    # ========================================================

    df["combined_risk_factors"] = (

        df.apply(

            combine_risk_factors,

            axis=1

        )

    )


    df["combined_risk_factor_count"] = (

        df["combined_risk_factors"]

        .apply(len)

    )


    # ========================================================
    # ADD ENGINE SUMMARY
    # ========================================================

    def get_engine_summary(row):

        engines = []


        if row["ml_detected"]:

            engines.append(
                "ML Anomaly"
            )


        if row["rule_detected"]:

            engines.append(
                "Rule-Based"
            )


        if row["financial_detected"]:

            engines.append(
                "Financial"
            )


        if row["statistical_detected"]:

            engines.append(
                "Statistical"
            )


        if len(engines) == 0:

            return "No risk engine detected significant anomaly"


        return ", ".join(engines)


    df["detecting_engines"] = (

        df.apply(

            get_engine_summary,

            axis=1

        )

    )


    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 60)

    print("FINAL UNIFIED AI RISK ANALYSIS V2 COMPLETE")

    print("=" * 60)


    # ========================================================
    # FINAL DISTRIBUTION
    # ========================================================

    print("\nFINAL RISK DISTRIBUTION\n")

    print(

        df["final_ai_risk_level"]

        .value_counts()

    )


    # ========================================================
    # SCORE STATISTICS
    # ========================================================

    print("\nFINAL AI RISK SCORE STATISTICS\n")

    print(

        df["final_ai_risk_score"]

        .describe()

    )


    # ========================================================
    # CONFIDENCE DISTRIBUTION
    # ========================================================

    print("\nRISK DETECTION CONFIDENCE\n")

    print(

        df["risk_detection_confidence"]

        .value_counts()

    )


    # ========================================================
    # ACTIVE ENGINE DISTRIBUTION
    # ========================================================

    print("\nMULTI-ENGINE DETECTION DISTRIBUTION\n")

    print(

        df["active_risk_engines"]

        .value_counts()

        .sort_index()

    )


    # ========================================================
    # CONSENSUS BONUS DISTRIBUTION
    # ========================================================

    print("\nCONSENSUS BONUS DISTRIBUTION\n")

    print(

        df["consensus_bonus"]

        .value_counts()

        .sort_index()

    )


    # ========================================================
    # PRIMARY RISK SOURCE
    # ========================================================

    print("\nPRIMARY RISK SOURCE DISTRIBUTION\n")

    print(

        df["primary_risk_source"]

        .value_counts()

    )


    # ========================================================
    # TOP RISK PROJECTS
    # ========================================================

    print("\nTOP UNIFIED AI RISK PROJECTS\n")


    display_columns = [

        "work_id"

    ]


    optional_columns = [

        "state",

        "ml_anomaly_score",

        "rule_risk_score",

        "financial_risk_score",

        "statistical_anomaly_score",

        "final_ai_risk_score",

        "final_ai_risk_level",

        "active_risk_engines",

        "consensus_bonus",

        "risk_detection_confidence",

        "primary_risk_source"

    ]


    for column in optional_columns:

        if column in df.columns:

            display_columns.append(
                column
            )


    print(

        df

        .sort_values(

            "final_ai_risk_score",

            ascending=False

        )

        [display_columns]

        .head(30)

    )


    # ========================================================
    # HIGH + CRITICAL PROJECTS
    # ========================================================

    high_risk_projects = df[

        df["final_ai_risk_level"]

        .isin([

            "HIGH",

            "CRITICAL"

        ])

    ]


    print(

        "\nHIGH + CRITICAL RISK PROJECTS:",

        len(high_risk_projects)

    )


    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(

        "data/processed",

        exist_ok=True

    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    df.to_csv(

        OUTPUT_PATH,

        index=False

    )


    print("\nResults saved successfully!")

    print(OUTPUT_PATH)


    return df


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    unified_df = build_unified_risk_engine()