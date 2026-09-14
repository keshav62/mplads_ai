import pandas as pd

from src.rule_engine import run_rule_engine


# ==========================================
# LOAD ENSEMBLE RESULTS
# ==========================================

df = pd.read_csv(

    "data/processed/ensemble_anomalies.csv"

)


# ==========================================
# RUN RULE ENGINE
# ==========================================

result_df = run_rule_engine(df)


# ==========================================
# SAVE RESULTS
# ==========================================

result_df.to_csv(

    "data/processed/rule_based_risk_analysis.csv",

    index=False

)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 60)

print("RULE-BASED AI RISK ANALYSIS COMPLETE")

print("=" * 60)


print("\nRULE RISK LEVEL DISTRIBUTION\n")

print(

    result_df["rule_risk_level"]
    .value_counts()

)


print("\nTOP HIGH-RISK PROJECTS\n")


columns = [

    "work_id",

    "state",

    "constituency",

    "rule_risk_score",

    "rule_risk_level",

    "risk_factors",

    "recommended_actions"

]


available_columns = [

    col for col in columns

    if col in result_df.columns

]


print(

    result_df
    .sort_values(

        by="rule_risk_score",

        ascending=False

    )
    [available_columns]
    .head(20)

)


print("\nResults saved:")

print(

    "data/processed/rule_based_risk_analysis.csv"

)

print("\nFINANCIAL RISK SCORE DISTRIBUTION\n")

