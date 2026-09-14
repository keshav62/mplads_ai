import pandas as pd

from src.statistical_anomaly_engine import (
    calculate_statistical_anomalies
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/processed/financial_risk_analysis.csv"
)


print(
    "\nLoaded dataset successfully"
)


print(
    "Total projects:",
    len(df)
)


# ============================================================
# CALCULATE STATISTICAL ANOMALIES
# ============================================================

result_df = calculate_statistical_anomalies(
    df
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)

print(
    "STATISTICAL FINANCIAL ANOMALY ANALYSIS COMPLETE"
)

print("=" * 60)


# ------------------------------------------------------------
# DISTRIBUTION
# ------------------------------------------------------------

print(
    "\nSTATISTICAL ANOMALY DISTRIBUTION\n"
)


print(
    result_df[
        "statistical_anomaly_level"
    ].value_counts()
)


# ------------------------------------------------------------
# SCORE STATISTICS
# ------------------------------------------------------------

print(
    "\nSTATISTICAL ANOMALY SCORE STATISTICS\n"
)


print(
    result_df[
        "statistical_anomaly_score"
    ].describe()
)


# ------------------------------------------------------------
# NON-ZERO ANOMALIES
# ------------------------------------------------------------

print(
    "\nNON-ZERO STATISTICAL ANOMALIES:"
)


print(
    (
        result_df[
            "statistical_anomaly_score"
        ] > 0
    ).sum()
)


# ------------------------------------------------------------
# TOP ANOMALIES
# ------------------------------------------------------------

print(
    "\nTOP STATISTICAL FINANCIAL ANOMALIES\n"
)


columns = [

    "work_id",

    "state",

    "total_expenditure",

    "payment_count",

    "unique_vendors",

    "payments_per_vendor",

    "statistical_anomaly_score",

    "statistical_anomaly_count",

    "statistical_anomaly_level",

    "statistical_anomaly_factors"

]


available_columns = [

    col for col in columns

    if col in result_df.columns

]


print(

    result_df

    .sort_values(

        by="statistical_anomaly_score",

        ascending=False

    )

    [available_columns]

    .head(20)

)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    "data/processed/"
    "statistical_financial_anomaly_analysis.csv"
)


result_df.to_csv(

    output_path,

    index=False

)


print(
    "\nResults saved successfully!"
)


print(
    output_path
)