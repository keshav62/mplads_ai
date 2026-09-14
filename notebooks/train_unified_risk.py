import pandas as pd

from src.unified_risk_engine import (
    calculate_unified_risk
)


# ============================================================
# LOAD RULE + ML DATASET
# ============================================================

print("\nLoading ML + rule-based dataset...")

rule_df = pd.read_csv(
    "data/processed/rule_based_risk_analysis.csv"
)

# The same dataset already contains ML anomaly columns
ml_df = rule_df.copy()

print(
    "Total projects:",
    len(rule_df)
)


# ============================================================
# LOAD FINANCIAL DATASET
# ============================================================

print("\nLoading financial risk dataset...")

financial_df = pd.read_csv(
    "data/processed/financial_risk_analysis.csv"
)

print(
    "Financial projects:",
    len(financial_df)
)


# ============================================================
# LOAD STATISTICAL DATASET
# ============================================================

print("\nLoading statistical anomaly dataset...")

statistical_df = pd.read_csv(
    "data/processed/statistical_financial_anomaly_analysis.csv"
)

print(
    "Statistical projects:",
    len(statistical_df)
)


# ============================================================
# SHOW AVAILABLE COLUMNS
# ============================================================

print("\nML + RULE DATASET COLUMNS:")

print(
    rule_df.columns.tolist()
)


# ============================================================
# CALCULATE UNIFIED RISK
# ============================================================

risk_df = calculate_unified_risk(

    ml_df=ml_df,

    rule_df=rule_df,

    financial_df=financial_df,

    statistical_df=statistical_df

)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    "data/processed/"
    "final_unified_risk_analysis.csv"
)

risk_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)

print(
    "FINAL UNIFIED AI RISK ANALYSIS COMPLETE"
)

print("=" * 60)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

print(
    "\nFINAL RISK DISTRIBUTION\n"
)

print(
    risk_df[
        "final_ai_risk_level"
    ].value_counts()
)


# ============================================================
# SCORE STATISTICS
# ============================================================

print(
    "\nFINAL AI RISK SCORE STATISTICS\n"
)

print(
    risk_df[
        "final_ai_risk_score"
    ].describe()
)


# ============================================================
# CONFIDENCE DISTRIBUTION
# ============================================================

print(
    "\nRISK DETECTION CONFIDENCE\n"
)

print(
    risk_df[
        "risk_detection_confidence"
    ].value_counts()
)


# ============================================================
# MULTI-ENGINE DETECTION
# ============================================================

print(
    "\nMULTI-ENGINE DETECTION DISTRIBUTION\n"
)

print(
    risk_df[
        "active_risk_engines"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# TOP RISK PROJECTS
# ============================================================

print(
    "\nTOP UNIFIED AI RISK PROJECTS\n"
)


columns = [

    "work_id",

    "ensemble_risk_level",

    "rule_risk_level",

    "financial_risk_level",

    "statistical_anomaly_level",

    "ml_normalized_score",

    "rule_risk_score",

    "financial_risk_score",

    "statistical_anomaly_score",

    "final_ai_risk_score",

    "final_ai_risk_level",

    "active_risk_engines",

    "risk_detection_confidence"

]


available_columns = [

    col for col in columns

    if col in risk_df.columns

]


print(

    risk_df

    .sort_values(
        by="final_ai_risk_score",
        ascending=False
    )

    [available_columns]

    .head(30)

)


# ============================================================
# HIGH AND CRITICAL PROJECTS
# ============================================================

print(
    "\nHIGH + CRITICAL RISK PROJECTS:"
)

high_risk = risk_df[

    risk_df[
        "final_ai_risk_level"
    ].isin([
        "HIGH",
        "CRITICAL"
    ])

]

print(
    len(high_risk)
)


# ============================================================
# SAVE MESSAGE
# ============================================================

print("\nResults saved successfully!")

print(output_path)