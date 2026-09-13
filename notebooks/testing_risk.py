import pandas as pd 

from src.risk_engine import calculate_risk_score

anomaly_df = pd.read_csv(
    "data/processed/financial_anomalies.csv"
)


risk_df = calculate_risk_score(
    anomaly_df
)

print("\nRISK LEVEL DISTRIBUTION")

print(
    risk_df["risk_level"].value_counts()
)

print("\nTOP HIGH-RISK PROJECTS")

print(
    risk_df
    .sort_values(
        "risk_score",
        ascending=False
    )
    [
        [
            "work_id",
            "state",
            "constituency",
            "risk_score",
            "risk_level",
            "anomaly_score"
        ]
    ]
    .head(20)
)

risk_df.to_csv(
    "data/processed/risk_scored_projects.csv",
    index=False
)

print(
    "\nRisk scored projects saved successfully!"
)