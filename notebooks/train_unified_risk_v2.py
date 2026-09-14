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

        print(
            f"WARNING: File not found: {path}"
        )

        return None

    df = pd.read_csv(path)

    print(
        f"Total projects: {len(df)}"
    )

    print("Columns available:")

    print(
        df.columns.tolist()
    )

    return df


# ============================================================
# FIND FIRST AVAILABLE FILE
# ============================================================

def load_first_available(
    paths,
    dataset_name
):

    print(
        f"\nLoading {dataset_name}..."
    )

    for path in paths:

        if os.path.exists(path):

            print(
                f"Using file: {path}"
            )

            df = pd.read_csv(path)

            print(
                f"Total projects: {len(df)}"
            )

            return df

    print(
        f"WARNING: No valid file found for {dataset_name}"
    )

    return None


# ============================================================
# SAFE NUMERIC CONVERSION
# ============================================================

def safe_numeric(
    df,
    column
):

    if column not in df.columns:

        df[column] = np.nan

        return df


    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


    df[column] = df[column].replace(
        [np.inf, -np.inf],
        np.nan
    )


    # IMPORTANT:
    # Do NOT fill missing engine scores with zero.
    #
    # NaN means the engine did not have a result
    # for that project.

    return df


# ============================================================
# NORMALIZE SCORE TO 0-100
# ============================================================

def normalize_score(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    valid_values = series.dropna()


    if len(valid_values) == 0:

        return pd.Series(
            np.nan,
            index=series.index
        )


    max_value = valid_values.max()


    if max_value <= 0:

        return pd.Series(
            0.0,
            index=series.index
        )


    normalized = (
        series / max_value
    ) * 100


    return normalized.clip(
        0,
        100
    )


# ============================================================
# RISK LEVEL TO NUMERIC SCORE
# ============================================================

def risk_level_to_score(level):

    if pd.isna(level):

        return np.nan


    level = str(level).upper()


    mapping = {

        "LOW": 0,

        "MEDIUM": 40,

        "HIGH": 70,

        "CRITICAL": 100

    }


    return mapping.get(
        level,
        np.nan
    )


# ============================================================
# GET FINAL RISK LEVEL
# ============================================================

def get_final_risk_level(
    score
):

    if pd.isna(score):

        return "INSUFFICIENT_DATA"


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

def get_risk_confidence(
    active_engines
):

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

def get_consensus_bonus(
    active_engines
):

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

    active_engines = int(
        row.get("active_risk_engines", 0)
    )

    # --------------------------------------------------------
    # No engine detected risk
    # --------------------------------------------------------

    if active_engines == 0:

        return "No Significant Risk"


    # --------------------------------------------------------
    # Multiple independent engines agree
    # --------------------------------------------------------
    #
    # If 2 or more engines independently detect risk,
    # classify the source as Multi-Engine Consensus.
    #
    # This is stronger than simply selecting the engine
    # with the highest numerical score.
    #

    if active_engines >= 2:

        return "Multi-Engine Consensus"


    # --------------------------------------------------------
    # Only one engine detected risk
    # --------------------------------------------------------

    if row.get("ml_detected", 0) == 1:

        return "ML Anomaly Engine"


    if row.get("rule_detected", 0) == 1:

        return "Rule-Based Risk Engine"


    if row.get("financial_detected", 0) == 1:

        return "Financial Risk Engine"


    if row.get("statistical_detected", 0) == 1:

        return "Statistical Anomaly Engine"


    return "No Significant Risk"


# ============================================================
# COMBINE RISK FACTORS
# ============================================================

def combine_risk_factors(
    row
):

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


        if value in [
            "[]",
            "",
            "nan",
            "None"
        ]:

            continue


        factors.append(
            value
        )


    # Remove duplicates while preserving order.

    unique_factors = []


    for factor in factors:

        if factor not in unique_factors:

            unique_factors.append(
                factor
            )


    return unique_factors


# ============================================================
# MAIN FUNCTION
# ============================================================

def build_unified_risk_engine():

    print(
        "\n" + "=" * 60
    )

    print(
        "UNIFIED RISK INTELLIGENCE ENGINE V2"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # LOAD MASTER DATASET
    # ========================================================

    master_df = load_csv_if_exists(

        MASTER_DATASET_PATH,

        "master dataset"

    )


    if master_df is None:

        raise FileNotFoundError(
            f"Master dataset not found: "
            f"{MASTER_DATASET_PATH}"
        )


    # ========================================================
    # CHECK WORK_ID
    # ========================================================

    if "work_id" not in master_df.columns:

        raise ValueError(
            "Master dataset must contain 'work_id'"
        )


    if not master_df["work_id"].is_unique:

        raise ValueError(
            "Master dataset contains duplicate work_id values."
        )


    # ========================================================
    # LOAD ML DATASET
    # ========================================================

    ml_df = load_first_available(

        [

            ML_DATASET_PATH,

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

            RULE_DATASET_PATH,

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

            FINANCIAL_DATASET_PATH,

            "data/processed/financial_anomalies.csv"

        ],

        "financial risk dataset"

    )


    # ========================================================
    # LOAD STATISTICAL DATASET
    # ========================================================

    statistical_df = load_first_available(

        [

            STATISTICAL_DATASET_PATH,

            "data/processed/statistical_anomaly_analysis.csv"

        ],

        "statistical anomaly dataset"

    )


    # ========================================================
    # START WITH MASTER DATASET
    # ========================================================

    df = master_df.copy()


    print(
        "\n" + "=" * 60
    )

    print(
        "MERGING RISK ENGINES"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # ML ENGINE
    # ========================================================

    if (
        ml_df is not None
        and
        "work_id" in ml_df.columns
    ):

        print(
            "\nUsing finalized ML ensemble output"
        )


        if not ml_df["work_id"].is_unique:

            raise ValueError(
                "ML ensemble contains duplicate work_id values."
            )


        # ----------------------------------------------------
        # SELECT FINAL ML SCORE
        # ----------------------------------------------------

        if (
            "ml_final_risk_score_100"
            in ml_df.columns
        ):

            ml_score_column = (
                "ml_final_risk_score_100"
            )

        elif (
            "ml_final_risk_score"
            in ml_df.columns
        ):

            ml_score_column = (
                "ml_final_risk_score"
            )

        elif (
            "combined_anomaly_score"
            in ml_df.columns
        ):

            ml_score_column = (
                "combined_anomaly_score"
            )

        else:

            raise ValueError(
                "Final ML risk score not found "
                "in ensemble dataset."
            )


        print(
            "ML risk score column:",
            ml_score_column
        )


        # ----------------------------------------------------
        # SELECT ML COLUMNS
        # ----------------------------------------------------

        ml_columns = [

            "work_id",

            ml_score_column

        ]


        if (
            "ensemble_is_anomaly"
            in ml_df.columns
        ):

            ml_columns.append(
                "ensemble_is_anomaly"
            )


        if (
            "both_models_anomaly"
            in ml_df.columns
        ):

            ml_columns.append(
                "both_models_anomaly"
            )


        if (
            "model_agreement"
            in ml_df.columns
        ):

            ml_columns.append(
                "model_agreement"
            )


        if (
            "primary_ml_source"
            in ml_df.columns
        ):

            ml_columns.append(
                "primary_ml_source"
            )


        if (
            "ensemble_risk_level"
            in ml_df.columns
        ):

            ml_columns.append(
                "ensemble_risk_level"
            )


        ml_merge = ml_df[
            ml_columns
        ].copy()


        # ----------------------------------------------------
        # RENAME ML SCORE
        # ----------------------------------------------------

        ml_merge = ml_merge.rename(

            columns={

                ml_score_column:
                    "ml_anomaly_score"

            }

        )


        # ----------------------------------------------------
        # CONVERT ML SCORE TO 0-100
        # ----------------------------------------------------

        if (
            ml_score_column
            ==
            "ml_final_risk_score"
        ):

            ml_merge[
                "ml_anomaly_score"
            ] = (

                pd.to_numeric(

                    ml_merge[
                        "ml_anomaly_score"
                    ],

                    errors="coerce"

                )

                * 100

            )


        elif (
            ml_score_column
            ==
            "combined_anomaly_score"
        ):

            ml_merge[
                "ml_anomaly_score"
            ] = (

                pd.to_numeric(

                    ml_merge[
                        "ml_anomaly_score"
                    ],

                    errors="coerce"

                )

                * 100

            )


        else:

            ml_merge[
                "ml_anomaly_score"
            ] = pd.to_numeric(

                ml_merge[
                    "ml_anomaly_score"
                ],

                errors="coerce"

            )


        # ----------------------------------------------------
        # MERGE ML
        # ----------------------------------------------------

        df = df.merge(

            ml_merge,

            on="work_id",

            how="left",

            validate="one_to_one"

        )


        # ----------------------------------------------------
        # ML AVAILABILITY
        # ----------------------------------------------------

        df["ml_available"] = (

            df["ml_anomaly_score"]
            .notna()
            .astype(int)

        )


        # ----------------------------------------------------
        # ACTUAL ML ENSEMBLE FLAG
        # ----------------------------------------------------

        if (
            "ensemble_is_anomaly"
            in df.columns
        ):

            df["ml_detected"] = (

                pd.to_numeric(

                    df[
                        "ensemble_is_anomaly"
                    ],

                    errors="coerce"

                )

                .fillna(0)

                .astype(int)

            )

        else:

            df["ml_detected"] = (

                df["ml_anomaly_score"]
                .fillna(0)
                >= 95

            ).astype(int)


    else:

        print(
            "\nWARNING: ML dataset unavailable"
        )

        df["ml_anomaly_score"] = np.nan

        df["ml_available"] = 0

        df["ml_detected"] = 0


    # ========================================================
    # RULE-BASED ENGINE
    # ========================================================

    if (
        rule_df is not None
        and
        "work_id" in rule_df.columns
    ):

        if not rule_df["work_id"].is_unique:

            raise ValueError(
                "Rule-based dataset contains duplicate work_id values."
            )


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
                "\nRule score column detected:",
                rule_score_column
            )


            columns_to_keep = [

                "work_id",

                rule_score_column

            ]


            if (
                "risk_factors"
                in rule_df.columns
            ):

                columns_to_keep.append(
                    "risk_factors"
                )


            rule_merge = rule_df[
                columns_to_keep
            ].copy()


            rule_merge = rule_merge.rename(

                columns={

                    rule_score_column:
                        "rule_risk_score"

                }

            )


            df = df.merge(

                rule_merge,

                on="work_id",

                how="left",

                validate="one_to_one"

            )


        else:

            print(
                "\nWARNING: No rule-based score column found"
            )

            df["rule_risk_score"] = np.nan


    else:

        print(
            "\nWARNING: Rule-based dataset unavailable"
        )

        df["rule_risk_score"] = np.nan


    # ========================================================
    # FINANCIAL ENGINE
    # ========================================================

    if (

        financial_df is not None

        and

        "work_id" in financial_df.columns

    ):

        if not financial_df[
            "work_id"
        ].is_unique:

            raise ValueError(
                "Financial dataset contains duplicate work_id values."
            )


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

            how="left",

            validate="one_to_one"

        )


    else:

        print(
            "\nWARNING: Financial dataset unavailable"
        )

        df["financial_risk_score"] = np.nan


    # ========================================================
    # STATISTICAL ENGINE
    # ========================================================

    if (

        statistical_df is not None

        and

        "work_id" in statistical_df.columns

    ):

        if not statistical_df[
            "work_id"
        ].is_unique:

            raise ValueError(
                "Statistical dataset contains duplicate work_id values."
            )


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

            how="left",

            validate="one_to_one"

        )


    else:

        print(
            "\nWARNING: Statistical dataset unavailable"
        )

        df[
            "statistical_anomaly_score"
        ] = np.nan


    # ========================================================
    # ENSURE SCORE COLUMNS EXIST
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
    # ENGINE AVAILABILITY
    # ========================================================

    df["ml_available"] = (
        df["ml_anomaly_score"]
        .notna()
        .astype(int)
    )


    df["rule_available"] = (
        df["rule_risk_score"]
        .notna()
        .astype(int)
    )


    df["financial_available"] = (
        df["financial_risk_score"]
        .notna()
        .astype(int)
    )


    df["statistical_available"] = (
        df["statistical_anomaly_score"]
        .notna()
        .astype(int)
    )


    # ========================================================
    # NORMALIZE ENGINE SCORES
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "PREPARING ENGINE RISK SCORES"
    )

    print(
        "=" * 60
    )


    # ML ensemble already produces 0-100.

    df["ml_normalized_score"] = (

        df["ml_anomaly_score"]
        .clip(0, 100)

    )


    # Other engines are normalized independently.

    df["rule_normalized_score"] = (
        normalize_score(
            df["rule_risk_score"]
        )
    )


    df["financial_normalized_score"] = (
        normalize_score(
            df["financial_risk_score"]
        )
    )


    df["statistical_normalized_score"] = (
        normalize_score(
            df["statistical_anomaly_score"]
        )
    )


    # ========================================================
    # RISK WEIGHTS
    # ========================================================

    ML_WEIGHT = 0.35

    RULE_WEIGHT = 0.25

    FINANCIAL_WEIGHT = 0.20

    STATISTICAL_WEIGHT = 0.20


    print(
        "\nRISK WEIGHTS"
    )


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
    # ENGINE DETECTION FLAGS
    # ========================================================

    RULE_THRESHOLD = 25

    FINANCIAL_THRESHOLD = 25

    STATISTICAL_THRESHOLD = 25


    # --------------------------------------------------------
    # ML
    # --------------------------------------------------------
    #
    # ML detection comes directly from the finalized
    # ensemble_is_anomaly flag.
    #

    if (
        "ensemble_is_anomaly"
        in df.columns
    ):

        df["ml_detected"] = (

            pd.to_numeric(

                df[
                    "ensemble_is_anomaly"
                ],

                errors="coerce"

            )

            .fillna(0)

            .astype(int)

        )


    # --------------------------------------------------------
    # RULE
    # --------------------------------------------------------

    df["rule_detected"] = np.where(

        (
            df["rule_available"] == 1
        )

        &

        (
            df[
                "rule_normalized_score"
            ]
            >= RULE_THRESHOLD
        ),

        1,

        0

    )


    # --------------------------------------------------------
    # FINANCIAL
    # --------------------------------------------------------

    df["financial_detected"] = np.where(

        (
            df[
                "financial_available"
            ] == 1
        )

        &

        (
            df[
                "financial_normalized_score"
            ]
            >= FINANCIAL_THRESHOLD
        ),

        1,

        0

    )


    # --------------------------------------------------------
    # STATISTICAL
    # --------------------------------------------------------

    df["statistical_detected"] = np.where(

        (
            df[
                "statistical_available"
            ] == 1
        )

        &

        (
            df[
                "statistical_normalized_score"
            ]
            >= STATISTICAL_THRESHOLD
        ),

        1,

        0

    )


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
    # AVAILABLE ENGINE COUNT
    # ========================================================

    df["available_risk_engines"] = (

        df["ml_available"]

        +

        df["rule_available"]

        +

        df["financial_available"]

        +

        df["statistical_available"]

    )


    # ========================================================
    # WEIGHTED AVAILABLE RISK
    # ========================================================

    weighted_score = (

        df[
            "ml_normalized_score"
        ].fillna(0)
        *
        ML_WEIGHT

        +

        df[
            "rule_normalized_score"
        ].fillna(0)
        *
        RULE_WEIGHT

        +

        df[
            "financial_normalized_score"
        ].fillna(0)
        *
        FINANCIAL_WEIGHT

        +

        df[
            "statistical_normalized_score"
        ].fillna(0)
        *
        STATISTICAL_WEIGHT

    )


    available_weight = (

        df["ml_available"]
        *
        ML_WEIGHT

        +

        df["rule_available"]
        *
        RULE_WEIGHT

        +

        df["financial_available"]
        *
        FINANCIAL_WEIGHT

        +

        df["statistical_available"]
        *
        STATISTICAL_WEIGHT

    )


    # Redistribute weights among engines that
    # actually have data.

    df["base_risk_score"] = np.where(

        available_weight > 0,

        weighted_score /
        available_weight,

        np.nan

    )


    df["base_risk_score"] = (

        df["base_risk_score"]
        .clip(0, 100)

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

        df[
            "final_ai_risk_score"
        ]

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

        df[
            "combined_risk_factors"
        ]

        .apply(len)

    )


    # ========================================================
    # ENGINE SUMMARY
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

            return (
                "No risk engine detected "
                "significant anomaly"
            )


        return ", ".join(
            engines
        )


    df["detecting_engines"] = (

        df.apply(

            get_engine_summary,

            axis=1

        )

    )


    # ========================================================
    # FINAL DATA QUALITY
    # ========================================================

    if len(df) != len(master_df):

        raise ValueError(

            "Unified engine changed "
            "the master row count."

        )


    if df["work_id"].nunique() != len(master_df):

        raise ValueError(

            "Duplicate work IDs detected "
            "in unified risk output."

        )


    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL UNIFIED AI RISK ANALYSIS V2 COMPLETE"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # FINAL DISTRIBUTION
    # ========================================================

    print(
        "\nFINAL RISK DISTRIBUTION\n"
    )


    print(

        df[
            "final_ai_risk_level"
        ]

        .value_counts()

    )


    # ========================================================
    # SCORE STATISTICS
    # ========================================================

    print(
        "\nFINAL AI RISK SCORE STATISTICS\n"
    )


    print(

        df[
            "final_ai_risk_score"
        ]

        .describe()

    )


    # ========================================================
    # CONFIDENCE DISTRIBUTION
    # ========================================================

    print(
        "\nRISK DETECTION CONFIDENCE\n"
    )


    print(

        df[
            "risk_detection_confidence"
        ]

        .value_counts()

    )


    # ========================================================
    # ACTIVE ENGINE DISTRIBUTION
    # ========================================================

    print(
        "\nMULTI-ENGINE DETECTION DISTRIBUTION\n"
    )


    print(

        df[
            "active_risk_engines"
        ]

        .value_counts()

        .sort_index()

    )


    # ========================================================
    # AVAILABLE ENGINE DISTRIBUTION
    # ========================================================

    print(
        "\nAVAILABLE ENGINE DISTRIBUTION\n"
    )


    print(

        df[
            "available_risk_engines"
        ]

        .value_counts()

        .sort_index()

    )


    # ========================================================
    # CONSENSUS BONUS DISTRIBUTION
    # ========================================================

    print(
        "\nCONSENSUS BONUS DISTRIBUTION\n"
    )


    print(

        df[
            "consensus_bonus"
        ]

        .value_counts()

        .sort_index()

    )


    # ========================================================
    # PRIMARY RISK SOURCE
    # ========================================================

    print(
        "\nPRIMARY RISK SOURCE DISTRIBUTION\n"
    )


    print(

        df[
            "primary_risk_source"
        ]

        .value_counts()

    )


    # ========================================================
    # TOP RISK PROJECTS
    # ========================================================

    print(
        "\nTOP UNIFIED AI RISK PROJECTS\n"
    )


    display_columns = [

        "work_id"

    ]


    optional_columns = [

        "state",

        "ml_anomaly_score",

        "ml_detected",

        "rule_risk_score",

        "rule_detected",

        "financial_risk_score",

        "financial_detected",

        "statistical_anomaly_score",

        "statistical_detected",

        "final_ai_risk_score",

        "final_ai_risk_level",

        "active_risk_engines",

        "available_risk_engines",

        "consensus_bonus",

        "risk_detection_confidence",

        "primary_risk_source",

        "detecting_engines"

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

        [

            display_columns

        ]

        .head(30)

    )


    # ========================================================
    # HIGH + CRITICAL PROJECTS
    # ========================================================

    high_risk_projects = df[

        df[
            "final_ai_risk_level"
        ]

        .isin(

            [

                "HIGH",

                "CRITICAL"

            ]

        )

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


    print(
        "\nResults saved successfully!"
    )

    print(
        OUTPUT_PATH
    )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print(
        "\nFINAL VALIDATION"
    )


    print(
        "Master projects:",
        len(master_df)
    )


    print(
        "Unified projects:",
        len(df)
    )


    print(
        "Unique work IDs:",
        df["work_id"].nunique()
    )


    if len(df) == len(master_df):

        print(
            "✓ PASS: All master projects preserved"
        )

    else:

        print(
            "✗ FAIL: Master projects not preserved"
        )


    if (
        df["work_id"].nunique()
        == len(master_df)
    ):

        print(
            "✓ PASS: All work IDs are unique"
        )

    else:

        print(
            "✗ FAIL: Duplicate work IDs detected"
        )


    print(
        "\n" + "=" * 60
    )

    print(
        "UNIFIED RISK ENGINE COMPLETE"
    )

    print(
        "=" * 60
    )


    return df


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    unified_df = (
        build_unified_risk_engine()
    )