import pandas as pd

from src.hybrid_risk_engine import (
    calculate_hybrid_risk
)


# ==========================================
# LOAD RULE-BASED / ENSEMBLE DATA
# ==========================================

df = pd.read_csv(
    "data/processed/rule_based_risk_analysis.csv"
)


print(
    "\nLoaded rule-based dataset successfully"
)

print(
    "Total projects:",
    len(df)
)


# ==========================================
# LOAD FINANCIAL RISK DATA
# ==========================================

financial_df = pd.read_csv(
    "data/processed/financial_risk_analysis.csv"
)


print(
    "\nLoaded financial risk dataset successfully"
)

print(
    "Total financial records:",
    len(financial_df)
)


# ==========================================
# SELECT FINANCIAL COLUMNS
# ==========================================

financial_columns = [

    "work_id",

    "recommended_amount",

    "sanction_amount",

    "total_expenditure",

    "expenditure_ratio",

    "financial_risk_score",

    "financial_risk_level",

    "financial_risk_factors"

]


# Keep only columns that exist

available_financial_columns = [

    col for col in financial_columns

    if col in financial_df.columns

]


financial_df = financial_df[
    available_financial_columns
]


# ==========================================
# MERGE FINANCIAL DATA
# ==========================================

df = df.merge(

    financial_df,

    on="work_id",

    how="left"

)


print(
    "\nDatasets merged successfully"
)


print(
    "Total projects after merge:",
    len(df)
)


# ==========================================
# HANDLE MISSING FINANCIAL VALUES
# ==========================================

if "financial_risk_score" in df.columns:

    df["financial_risk_score"] = (

        df["financial_risk_score"]

        .fillna(0)

    )


if "financial_risk_level" in df.columns:

    df["financial_risk_level"] = (

        df["financial_risk_level"]

        .fillna("LOW")

    )


if "financial_risk_factors" in df.columns:

    df["financial_risk_factors"] = (

        df["financial_risk_factors"]

        .fillna("[]")

    )


# ==========================================
# SHOW AVAILABLE COLUMNS
# ==========================================

print(
    "\nAVAILABLE COLUMNS IN HYBRID DATASET\n"
)

print(
    df.columns.tolist()
)


# ==========================================
# CALCULATE HYBRID AI RISK
# ==========================================

risk_df = calculate_hybrid_risk(
    df
)


# ==========================================
# SAVE RESULTS
# ==========================================

risk_df.to_csv(

    "data/processed/final_ai_risk_analysis.csv",

    index=False

)


# ==========================================
# RESULTS
# ==========================================

print("\n" + "=" * 60)

print(
    "HYBRID AI RISK ANALYSIS COMPLETE"
)

print("=" * 60)


# ------------------------------------------
# RISK DISTRIBUTION
# ------------------------------------------

print(
    "\nFINAL AI RISK DISTRIBUTION\n"
)


print(

    risk_df[
        "final_ai_risk_level"
    ].value_counts()

)


# ------------------------------------------
# TOP AI RISK PROJECTS
# ------------------------------------------

print(
    "\nTOP AI HIGH-RISK PROJECTS\n"
)


columns = [

    "work_id",

    "state",

    "constituency",

    "ensemble_risk_level",

    "rule_risk_level",

    "financial_risk_score",

    "financial_risk_level",

    "final_ai_risk_score",

    "final_ai_risk_level"

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

    .head(20)

)


# ------------------------------------------
# TOP FINANCIAL RISK PROJECTS
# ------------------------------------------

print(
    "\nTOP FINANCIAL RISK PROJECTS\n"
)


financial_display_columns = [

    "work_id",

    "state",

    "recommended_amount",

    "sanction_amount",

    "total_expenditure",

    "expenditure_ratio",

    "financial_risk_score",

    "financial_risk_level"

]


available_financial_display_columns = [

    col for col in financial_display_columns

    if col in risk_df.columns

]


if (
    "financial_risk_score"
    in risk_df.columns
):

    print(

        risk_df

        .sort_values(

            by="financial_risk_score",

            ascending=False

        )

        [available_financial_display_columns]

        .head(20)

    )

else:

    print(
        "Financial risk score not available."
    )


# ==========================================
# SUCCESS MESSAGE
# ==========================================

print(
    "\nResults saved successfully!"
)

print(
    "data/processed/final_ai_risk_analysis.csv"
)