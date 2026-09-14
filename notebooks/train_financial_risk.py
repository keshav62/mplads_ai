import pandas as pd

from src.financial_risk_engine import (
    calculate_financial_risk
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(

    "data/processed/master_dataset.csv"

)


print(

    "\nLoaded dataset successfully"

)


print(

    "Total projects:",

    len(df)

)


# ============================================================
# CALCULATE FINANCIAL RISK
# ============================================================

risk_df = calculate_financial_risk(df)


# ============================================================
# SAVE RESULTS
# ============================================================

risk_df.to_csv(

    "data/processed/financial_risk_analysis.csv",

    index=False

)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)

print(

    "FINANCIAL RISK ANALYSIS COMPLETE"

)

print("=" * 60)


print(

    "\nFINANCIAL RISK DISTRIBUTION\n"

)


print(

    risk_df[
        "financial_risk_level"
    ].value_counts()

)


print(

    "\nFINANCIAL RISK SCORE STATISTICS\n"

)


print(

    risk_df[
        "financial_risk_score"
    ].describe()

)


print(

    "\nNON-ZERO FINANCIAL RISK PROJECTS:",

    (

        risk_df[
            "financial_risk_score"
        ] > 0

    ).sum()

)


# ============================================================
# TOP FINANCIAL RISK PROJECTS
# ============================================================

print(

    "\nTOP FINANCIAL RISK PROJECTS\n"

)


columns = [

    "work_id",

    "state",

    "recommended_amount",

    "sanction_amount",

    "total_expenditure",

    "expenditure_vs_sanction",

    "vendor_concentration",

    "financial_risk_score",

    "financial_risk_level",

    "financial_risk_factors"

]


available_columns = [

    col for col in columns

    if col in risk_df.columns

]


print(

    risk_df

    .sort_values(

        by="financial_risk_score",

        ascending=False

    )

    [available_columns]

    .head(20)

)


print(

    "\nResults saved successfully!"

)

print(

    "data/processed/financial_risk_analysis.csv"

)

print("\nFINANCIAL RISK FACTOR FREQUENCY\n")

factor_counts = {}

# Use the dataframe returned by the financial risk engine
for factors in risk_df["financial_risk_factors"]:

    if isinstance(factors, list):

        for factor in factors:

            factor_counts[factor] = (
                factor_counts.get(factor, 0) + 1
            )


factor_counts = dict(

    sorted(

        factor_counts.items(),

        key=lambda x: x[1],

        reverse=True

    )

)


for factor, count in factor_counts.items():

    print(

        f"{factor}: {count}"

    )