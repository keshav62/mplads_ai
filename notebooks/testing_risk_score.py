import pandas as pd

from src.risk_engine import calculate_risk_score
from src.risk_reason_engine import generate_risk_reasons


anomaly_df = pd.read_csv(
    "data/processed/financial_anomalies.csv"
)


# Calculate risk score
risk_df = calculate_risk_score(
    anomaly_df
)


# Generate explanations
risk_df = generate_risk_reasons(
    risk_df
)


print("\nRISK PROJECTS WITH REASONS\n")


print(
    risk_df[
        [
            "work_id",
            "state",
            "constituency",
            "risk_score",
            "risk_level",
            "risk_reasons"
        ]
    ]
    .sort_values(
        "risk_score",
        ascending=False
    )
    .head(20)
)


# Save final results

risk_df.to_csv(
    "data/processed/risk_scored_projects.csv",
    index=False
)


print(
    "\nRisk scored projects with explanations saved successfully!"
)

import pandas as pd

pd.set_option(
    "display.max_colwidth",
    None
)

print(
    risk_df[
        [
            "work_id",
            "risk_score",
            "risk_level",
            "risk_reasons"
        ]
    ]
    .sort_values(
        "risk_score",
        ascending=False
    )
    .head(20)
)