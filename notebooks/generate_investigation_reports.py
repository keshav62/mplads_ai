import pandas as pd

from src.explainability.investigation_engine import (
    generate_investigation_reports
)


print("\nLoading unified risk analysis...")

df = pd.read_csv(
    "data/processed/final_unified_risk_analysis_v2.csv"
)


print(
    f"Total projects: {len(df)}"
)


# =====================================================
# GENERATE REPORTS
# =====================================================

df = generate_investigation_reports(df)


# =====================================================
# DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 60)

print("AI INVESTIGATION REPORT GENERATION COMPLETE")

print("=" * 60)


print("\nTOP HIGH-RISK PROJECTS\n")


high_risk = df[
    df["final_ai_risk_level"].isin(
        ["HIGH", "CRITICAL"]
    )
]


columns_to_show = [

    "work_id",

    "state",

    "final_ai_risk_score",

    "final_ai_risk_level",

    "risk_detection_confidence",

    "active_risk_engines",

    "primary_risk_source",

    "risk_explanation",

    "recommended_actions"

]


available_columns = [

    col

    for col in columns_to_show

    if col in df.columns

]


print(

    high_risk[
        available_columns
    ]

    .sort_values(
        by="final_ai_risk_score",
        ascending=False
    )

    .head(20)

)


# =====================================================
# SAVE RESULTS
# =====================================================

output_path = (

    "data/processed/"
    "project_investigation_reports.csv"

)


df.to_csv(

    output_path,

    index=False

)


print("\nResults saved successfully!")

print(output_path)