"""
evaluate_risk_models.py

Evaluation framework for MPLADS AI Risk Intelligence System.

Evaluates:
    1. Isolation Forest / ML anomaly engine
    2. Rule-based risk engine
    3. Financial risk engine
    4. Statistical anomaly engine
    5. Unified AI risk engine

IMPORTANT:
This project currently uses unsupervised anomaly detection.
Therefore, accuracy/precision/recall are only calculated when
verified ground-truth labels are available.

Otherwise the script performs:
    - data integrity checks
    - engine coverage
    - engine agreement
    - score statistics
    - top-K overlap
    - risk distribution
    - ranking diagnostics
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

MASTER_FILE = os.path.join(DATA_DIR, "master_dataset.csv")
ML_FILE = os.path.join(DATA_DIR, "ensemble_anomalies.csv")
RULE_FILE = os.path.join(DATA_DIR, "rule_based_risk_analysis.csv")
FINANCIAL_FILE = os.path.join(DATA_DIR, "financial_risk_analysis.csv")
STATISTICAL_FILE = os.path.join(
    DATA_DIR,
    "statistical_financial_anomaly_analysis.csv"
)
UNIFIED_FILE = os.path.join(
    DATA_DIR,
    "final_unified_risk_analysis_v2.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "model_evaluation_summary.csv"
)

DETAILED_OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "model_evaluation_detailed.csv"
)


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_read_csv(path, name):
    if not os.path.exists(path):
        print(f"[WARNING] {name} not found:")
        print(path)
        return None

    try:
        df = pd.read_csv(path)
        print(f"[OK] {name}: {len(df):,} rows")
        return df

    except Exception as e:
        print(f"[ERROR] Could not load {name}")
        print(e)
        return None


def find_column(df, candidates):

    if df is None:
        return None

    for col in candidates:
        if col in df.columns:
            return col

    return None


def numeric_series(df, column):

    if column is None or column not in df.columns:
        return pd.Series(dtype=float)

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# LOAD DATASETS
# ============================================================

print_header("LOADING DATASETS")

master = safe_read_csv(
    MASTER_FILE,
    "Master Dataset"
)

ml = safe_read_csv(
    ML_FILE,
    "ML Anomaly Dataset"
)

rule = safe_read_csv(
    RULE_FILE,
    "Rule-Based Risk Dataset"
)

financial = safe_read_csv(
    FINANCIAL_FILE,
    "Financial Risk Dataset"
)

statistical = safe_read_csv(
    STATISTICAL_FILE,
    "Statistical Risk Dataset"
)

unified = safe_read_csv(
    UNIFIED_FILE,
    "Unified AI Risk Dataset"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print_header("BASIC DATASET VALIDATION")

if master is None:
    raise FileNotFoundError(
        "master_dataset.csv is required."
    )

if "work_id" not in master.columns:
    raise ValueError(
        "master_dataset.csv does not contain work_id."
    )

print(f"Master projects: {len(master):,}")

print(
    f"Unique work IDs: "
    f"{master['work_id'].nunique():,}"
)

duplicate_master = (
    master["work_id"].duplicated().sum()
)

print(
    f"Duplicate work IDs: "
    f"{duplicate_master:,}"
)


# ============================================================
# MERGE ALL ENGINES
# ============================================================

print_header("MERGING AI ENGINES")

evaluation = master.copy()

datasets = [
    ("ML", ml),
    ("RULE", rule),
    ("FINANCIAL", financial),
    ("STATISTICAL", statistical),
    ("UNIFIED", unified),
]

for name, df in datasets:

    if df is None:
        continue

    if "work_id" not in df.columns:
        print(
            f"[WARNING] {name} dataset has no work_id. "
            f"Skipping."
        )
        continue

    # Prevent duplicate work_id from exploding rows
    df = df.drop_duplicates(
        subset=["work_id"]
    )

    before = len(evaluation)

    evaluation = evaluation.merge(
        df,
        on="work_id",
        how="left",
        suffixes=("", f"_{name.lower()}")
    )

    after = len(evaluation)

    print(
        f"{name:12s} | "
        f"before={before:,} | "
        f"after={after:,}"
    )


print(
    f"\nFinal evaluation rows: "
    f"{len(evaluation):,}"
)


if len(evaluation) != len(master):
    print(
        "[WARNING] Row count changed during merge!"
    )
else:
    print(
        "[OK] Row count preserved."
    )


# ============================================================
# FIND IMPORTANT COLUMNS
# ============================================================

print_header("DETECTING MODEL COLUMNS")


ml_score_col = find_column(
    evaluation,
    [
        "combined_anomaly_score",
        "if_risk_score",
        "anomaly_score"
    ]
)

ml_flag_col = find_column(
    evaluation,
    [
        "ensemble_is_anomaly",
        "both_models_anomaly",
        "any_model_anomaly",
        "is_anomaly"
    ]
)

rule_score_col = find_column(
    evaluation,
    [
        "rule_risk_score"
    ]
)

rule_level_col = find_column(
    evaluation,
    [
        "rule_risk_level"
    ]
)

financial_score_col = find_column(
    evaluation,
    [
        "financial_risk_score"
    ]
)

financial_level_col = find_column(
    evaluation,
    [
        "financial_risk_level"
    ]
)

stat_score_col = find_column(
    evaluation,
    [
        "statistical_anomaly_score",
        "statistical_risk_score",
        "anomaly_score"
    ]
)

stat_level_col = find_column(
    evaluation,
    [
        "statistical_anomaly_level",
        "statistical_risk_level"
    ]
)

unified_score_col = find_column(
    evaluation,
    [
        "final_ai_risk_score"
    ]
)

unified_level_col = find_column(
    evaluation,
    [
        "final_ai_risk_level"
    ]
)

confidence_col = find_column(
    evaluation,
    [
        "risk_detection_confidence"
    ]
)

active_engines_col = find_column(
    evaluation,
    [
        "active_risk_engines"
    ]
)


columns_found = {
    "ML score": ml_score_col,
    "ML flag": ml_flag_col,
    "Rule score": rule_score_col,
    "Rule level": rule_level_col,
    "Financial score": financial_score_col,
    "Financial level": financial_level_col,
    "Statistical score": stat_score_col,
    "Statistical level": stat_level_col,
    "Unified score": unified_score_col,
    "Unified level": unified_level_col,
    "Confidence": confidence_col,
    "Active engines": active_engines_col,
}

for name, col in columns_found.items():

    if col:
        print(f"[OK] {name:20s}: {col}")

    else:
        print(f"[--] {name:20s}: NOT FOUND")


# ============================================================
# MODEL COVERAGE
# ============================================================

print_header("MODEL COVERAGE")

results = []

def add_result(
    model,
    metric,
    value,
    interpretation=""
):

    results.append({
        "model": model,
        "metric": metric,
        "value": value,
        "interpretation": interpretation
    })


models = [
    ("ML", ml_score_col),
    ("Rule", rule_score_col),
    ("Financial", financial_score_col),
    ("Statistical", stat_score_col),
    ("Unified", unified_score_col),
]

for model_name, score_col in models:

    if score_col is None:
        continue

    s = numeric_series(
        evaluation,
        score_col
    )

    available = s.notna().sum()

    coverage = (
        available / len(evaluation) * 100
    )

    print(
        f"{model_name:15s}: "
        f"{available:,}/{len(evaluation):,} "
        f"({coverage:.2f}%)"
    )

    add_result(
        model_name,
        "coverage_percent",
        round(coverage, 4),
        "Percentage of projects with a model score"
    )


# ============================================================
# SCORE STATISTICS
# ============================================================

print_header("MODEL SCORE STATISTICS")


for model_name, score_col in models:

    if score_col is None:
        continue

    s = numeric_series(
        evaluation,
        score_col
    ).dropna()

    if len(s) == 0:
        continue

    print(f"\n{model_name}")

    print(
        f"  Mean   : {s.mean():.4f}"
    )

    print(
        f"  Median : {s.median():.4f}"
    )

    print(
        f"  Std    : {s.std():.4f}"
    )

    print(
        f"  Min    : {s.min():.4f}"
    )

    print(
        f"  Max    : {s.max():.4f}"
    )

    print(
        f"  95%    : {s.quantile(.95):.4f}"
    )

    print(
        f"  99%    : {s.quantile(.99):.4f}"
    )

    add_result(
        model_name,
        "mean_score",
        round(s.mean(), 6)
    )

    add_result(
        model_name,
        "median_score",
        round(s.median(), 6)
    )

    add_result(
        model_name,
        "max_score",
        round(s.max(), 6)
    )


# ============================================================
# RISK DISTRIBUTIONS
# ============================================================

print_header("RISK LEVEL DISTRIBUTIONS")


level_columns = [
    ("Rule", rule_level_col),
    ("Financial", financial_level_col),
    ("Statistical", stat_level_col),
    ("Unified", unified_level_col),
]

for model_name, level_col in level_columns:

    if level_col is None:
        continue

    print(f"\n{model_name}")

    distribution = (
        evaluation[level_col]
        .fillna("MISSING")
        .value_counts()
    )

    print(distribution)

    for level, count in distribution.items():

        percentage = (
            count / len(evaluation) * 100
        )

        add_result(
            model_name,
            f"risk_level_{level}_percent",
            round(percentage, 4)
        )


# ============================================================
# HIGH + CRITICAL DETECTION
# ============================================================

print_header("HIGH / CRITICAL DETECTION")

for model_name, level_col in level_columns:

    if level_col is None:
        continue

    levels = (
        evaluation[level_col]
        .astype(str)
        .str.upper()
    )

    high_count = levels.isin(
        ["HIGH", "CRITICAL"]
    ).sum()

    critical_count = (
        levels == "CRITICAL"
    ).sum()

    print(
        f"{model_name:15s}: "
        f"HIGH+CRITICAL = {high_count:,} | "
        f"CRITICAL = {critical_count:,}"
    )

    add_result(
        model_name,
        "high_critical_count",
        int(high_count)
    )

    add_result(
        model_name,
        "critical_count",
        int(critical_count)
    )


# ============================================================
# ENGINE AGREEMENT
# ============================================================

print_header("ENGINE AGREEMENT")


flag_columns = []

if ml_flag_col:
    ml_flags = (
        evaluation[ml_flag_col]
        .fillna(False)
        .astype(bool)
    )

    evaluation["_ml_flag"] = ml_flags
    flag_columns.append("_ml_flag")


if rule_level_col:
    evaluation["_rule_flag"] = (
        evaluation[rule_level_col]
        .astype(str)
        .str.upper()
        .isin(["HIGH", "CRITICAL"])
    )

    flag_columns.append("_rule_flag")


if financial_level_col:
    evaluation["_financial_flag"] = (
        evaluation[financial_level_col]
        .astype(str)
        .str.upper()
        .isin(["HIGH", "CRITICAL"])
    )

    flag_columns.append("_financial_flag")


if stat_level_col:
    evaluation["_statistical_flag"] = (
        evaluation[stat_level_col]
        .astype(str)
        .str.upper()
        .isin(["HIGH", "CRITICAL"])
    )

    flag_columns.append("_statistical_flag")


if len(flag_columns) >= 2:

    evaluation["_engine_agreement_count"] = (
        evaluation[flag_columns]
        .sum(axis=1)
    )

    agreement_distribution = (
        evaluation["_engine_agreement_count"]
        .value_counts()
        .sort_index()
    )

    print(
        "Number of engines flagging each project:"
    )

    print(
        agreement_distribution
    )

    for count, number_projects in (
        agreement_distribution.items()
    ):

        percentage = (
            number_projects /
            len(evaluation) * 100
        )

        add_result(
            "Ensemble",
            f"{int(count)}_engine_agreement_percent",
            round(percentage, 4)
        )


# ============================================================
# TOP-K ANALYSIS
# ============================================================

print_header("TOP-K RISK ANALYSIS")


score_columns = {
    "ML": ml_score_col,
    "Rule": rule_score_col,
    "Financial": financial_score_col,
    "Statistical": stat_score_col,
    "Unified": unified_score_col,
}

top_k_values = [
    10,
    20,
    50,
    100
]

top_sets = {}

for model_name, score_col in score_columns.items():

    if score_col is None:
        continue

    temp = evaluation[
        ["work_id", score_col]
    ].copy()

    temp[score_col] = pd.to_numeric(
        temp[score_col],
        errors="coerce"
    )

    temp = temp.dropna(
        subset=[score_col]
    )

    temp = temp.sort_values(
        score_col,
        ascending=False
    )

    top_sets[model_name] = {}

    print(f"\n{model_name}")

    for k in top_k_values:

        ids = set(
            temp.head(k)["work_id"]
        )

        top_sets[model_name][k] = ids

        print(
            f"  Top-{k}: {len(ids)} projects"
        )


# ============================================================
# TOP-K OVERLAP WITH UNIFIED
# ============================================================

if "Unified" in top_sets:

    print_header(
        "TOP-K OVERLAP WITH UNIFIED AI MODEL"
    )

    unified_sets = top_sets["Unified"]

    for model_name, model_sets in top_sets.items():

        if model_name == "Unified":
            continue

        print(f"\n{model_name}")

        for k in top_k_values:

            if k not in model_sets:
                continue

            overlap = (
                model_sets[k]
                .intersection(
                    unified_sets.get(k, set())
                )
            )

            overlap_percent = (
                len(overlap) / k * 100
            )

            print(
                f"  Top-{k} overlap: "
                f"{len(overlap)}/{k} "
                f"({overlap_percent:.2f}%)"
            )

            add_result(
                model_name,
                f"top_{k}_overlap_with_unified_percent",
                round(overlap_percent, 4)
            )


# ============================================================
# UNIFIED SCORE CONCENTRATION
# ============================================================

if unified_score_col:

    print_header(
        "UNIFIED MODEL SCORE CONCENTRATION"
    )

    scores = numeric_series(
        evaluation,
        unified_score_col
    ).fillna(0)

    zero_count = (
        scores == 0
    ).sum()

    nonzero_count = (
        scores > 0
    ).sum()

    print(
        f"Zero scores     : {zero_count:,}"
    )

    print(
        f"Non-zero scores : {nonzero_count:,}"
    )

    print(
        f"Non-zero rate   : "
        f"{nonzero_count / len(scores) * 100:.2f}%"
    )

    add_result(
        "Unified",
        "zero_score_percent",
        round(
            zero_count /
            len(scores) *
            100,
            4
        )
    )


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

if confidence_col:

    print_header(
        "RISK DETECTION CONFIDENCE"
    )

    confidence_distribution = (
        evaluation[confidence_col]
        .fillna("MISSING")
        .value_counts()
    )

    print(
        confidence_distribution
    )

    for level, count in (
        confidence_distribution.items()
    ):

        percentage = (
            count /
            len(evaluation) *
            100
        )

        add_result(
            "Unified",
            f"confidence_{level}_percent",
            round(percentage, 4)
        )


# ============================================================
# ACTIVE ENGINE ANALYSIS
# ============================================================

if active_engines_col:

    print_header(
        "ACTIVE RISK ENGINE ANALYSIS"
    )

    active = pd.to_numeric(
        evaluation[active_engines_col],
        errors="coerce"
    )

    distribution = (
        active
        .fillna(0)
        .astype(int)
        .value_counts()
        .sort_index()
    )

    print(distribution)

    for engines, count in distribution.items():

        percentage = (
            count /
            len(evaluation) *
            100
        )

        add_result(
            "Unified",
            f"{int(engines)}_active_engines_percent",
            round(percentage, 4)
        )


# ============================================================
# GROUND TRUTH SEARCH
# ============================================================

print_header(
    "GROUND-TRUTH EVALUATION"
)


ground_truth_col = find_column(
    evaluation,
    [
        "actual_risk",
        "ground_truth",
        "audit_verified_risk",
        "verified_risk",
        "audit_label",
        "true_label"
    ]
)


if ground_truth_col is None:

    print(
        "[INFO] No verified ground-truth label found."
    )

    print(
        "\nTherefore:"
    )

    print(
        "  Accuracy  : NOT AVAILABLE"
    )

    print(
        "  Precision : NOT AVAILABLE"
    )

    print(
        "  Recall    : NOT AVAILABLE"
    )

    print(
        "  F1-score  : NOT AVAILABLE"
    )

    print(
        "\nThis is expected because the current"
        " anomaly models are unsupervised."
    )

    print(
        "\nRecommended next step:"
    )

    print(
        "Create an audit_verified_risk column "
        "for a manually reviewed sample."
    )

else:

    print(
        f"[OK] Ground truth found: "
        f"{ground_truth_col}"
    )

    print(
        evaluation[ground_truth_col]
        .value_counts(dropna=False)
    )

    # sklearn imported only when required
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report
    )

    if unified_level_col:

        valid = evaluation[
            [ground_truth_col, unified_level_col]
        ].dropna()

        if len(valid) > 0:

            y_true = (
                valid[ground_truth_col]
                .astype(str)
                .str.upper()
            )

            y_pred = (
                valid[unified_level_col]
                .astype(str)
                .str.upper()
            )

            # Binary evaluation:
            # LOW/MEDIUM = 0
            # HIGH/CRITICAL = 1

            y_true_binary = (
                y_true
                .isin(["HIGH", "CRITICAL"])
                .astype(int)
            )

            y_pred_binary = (
                y_pred
                .isin(["HIGH", "CRITICAL"])
                .astype(int)
            )

            accuracy = accuracy_score(
                y_true_binary,
                y_pred_binary
            )

            precision = precision_score(
                y_true_binary,
                y_pred_binary,
                zero_division=0
            )

            recall = recall_score(
                y_true_binary,
                y_pred_binary,
                zero_division=0
            )

            f1 = f1_score(
                y_true_binary,
                y_pred_binary,
                zero_division=0
            )

            print(
                f"\nUnified AI Classification"
            )

            print(
                f"Accuracy  : {accuracy:.4f}"
            )

            print(
                f"Precision : {precision:.4f}"
            )

            print(
                f"Recall    : {recall:.4f}"
            )

            print(
                f"F1 Score  : {f1:.4f}"
            )

            print(
                "\nConfusion Matrix:"
            )

            print(
                confusion_matrix(
                    y_true_binary,
                    y_pred_binary
                )
            )

            print(
                "\nClassification Report:"
            )

            print(
                classification_report(
                    y_true_binary,
                    y_pred_binary,
                    zero_division=0
                )
            )

            add_result(
                "Unified",
                "accuracy",
                round(accuracy, 6)
            )

            add_result(
                "Unified",
                "precision",
                round(precision, 6)
            )

            add_result(
                "Unified",
                "recall",
                round(recall, 6)
            )

            add_result(
                "Unified",
                "f1_score",
                round(f1, 6)
            )


# ============================================================
# HIGH-RISK PROJECT SANITY CHECK
# ============================================================

print_header(
    "HIGH-RISK PROJECT SANITY CHECK"
)

if unified_level_col:

    high_risk = evaluation[
        evaluation[unified_level_col]
        .astype(str)
        .str.upper()
        .isin(["HIGH", "CRITICAL"])
    ].copy()

    print(
        f"High + Critical projects: "
        f"{len(high_risk):,}"
    )

    if len(high_risk) > 0:

        useful_columns = [
            "work_id",
            "state",
            "constituency",
            "work_category",
            "total_expenditure",
            "payment_count",
            "unique_vendors",
            unified_score_col,
            unified_level_col
        ]

        useful_columns = [
            c for c in useful_columns
            if c is not None and c in high_risk.columns
        ]

        print(
            "\nTop high-risk projects:"
        )

        print(
            high_risk
            .sort_values(
                unified_score_col,
                ascending=False
            )[useful_columns]
            .head(20)
            .to_string(index=False)
        )


# ============================================================
# SAVE RESULTS
# ============================================================

print_header(
    "SAVING EVALUATION RESULTS"
)


summary_df = pd.DataFrame(
    results
)

summary_df.to_csv(
    OUTPUT_FILE,
    index=False
)

evaluation.to_csv(
    DETAILED_OUTPUT_FILE,
    index=False
)

print(
    f"[OK] Summary saved:"
)

print(
    OUTPUT_FILE
)

print(
    f"[OK] Detailed evaluation saved:"
)

print(
    DETAILED_OUTPUT_FILE
)


# ============================================================
# FINAL REPORT
# ============================================================

print_header(
    "AI MODEL EVALUATION COMPLETE"
)

print(
    f"Projects evaluated : "
    f"{len(evaluation):,}"
)

if unified_level_col:

    levels = (
        evaluation[unified_level_col]
        .astype(str)
        .str.upper()
    )

    print(
        f"HIGH + CRITICAL    : "
        f"{levels.isin(['HIGH', 'CRITICAL']).sum():,}"
    )

    print(
        f"CRITICAL           : "
        f"{(levels == 'CRITICAL').sum():,}"
    )

if unified_score_col:

    scores = numeric_series(
        evaluation,
        unified_score_col
    )

    print(
        f"Maximum AI score   : "
        f"{scores.max():.4f}"
    )

print(
    "\nIMPORTANT:"
)

print(
    "This evaluation does NOT claim model accuracy "
    "without verified audit labels."
)

print(
    "For true accuracy, create a ground-truth "
    "audit dataset and rerun this evaluator."
)

print(
    "\nDone."
)