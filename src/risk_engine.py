import pandas as pd
import numpy as np


def calculate_risk_score(df):

    risk_df = df.copy()

    # Start with zero risk
    risk_df["risk_score"] = 0


    # -----------------------------------
    # 1. ANOMALY RISK
    # -----------------------------------

    risk_df.loc[
        risk_df["is_anomaly"] == 1,
        "risk_score"
    ] += 40
    


    # -----------------------------------
    # 2. EXPENDITURE RATIO RISK
    # -----------------------------------

    risk_df.loc[
        risk_df["expenditure_ratio"] < 0.50,
        "risk_score"
    ] += 20


    # -----------------------------------
    # 3. HIGH PAYMENT COUNT RISK
    # -----------------------------------

    risk_df.loc[
        risk_df["payment_count"] > 5,
        "risk_score"
    ] += 15


    # -----------------------------------
    # 4. HIGH NUMBER OF VENDORS
    # -----------------------------------

    risk_df.loc[
        risk_df["unique_vendors"] > 5,
        "risk_score"
    ] += 15


    # -----------------------------------
    # 5. HIGH PAYMENTS PER VENDOR
    # -----------------------------------

    risk_df.loc[
        risk_df["payments_per_vendor"] > 3,
        "risk_score"
    ] += 10


    # Make sure score stays between 0 and 100

    risk_df["risk_score"] = risk_df[
        "risk_score"
    ].clip(0, 100)


    # -----------------------------------
    # CREATE RISK LEVELS
    # -----------------------------------

    def get_risk_level(score):

        if score >= 70:
            return "HIGH"

        elif score >= 40:
            return "MEDIUM"

        return "LOW"


    risk_df["risk_level"] = risk_df[
        "risk_score"
    ].apply(get_risk_level)


    return risk_df