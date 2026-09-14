import pandas as pd
import numpy as np


# ==========================================
# HYBRID AI RISK SCORING ENGINE V2
# ==========================================

def calculate_hybrid_risk(df):

    risk_df = df.copy()

    # ==========================================
    # ENSURE REQUIRED COLUMNS EXIST
    # ==========================================

    required_columns = [
        "combined_anomaly_score",
        "ensemble_risk_level",
        "rule_risk_level"
    ]

    missing_columns = [

        col for col in required_columns

        if col not in risk_df.columns

    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


    # ==========================================
    # 1. ENSEMBLE AI SCORE
    # ==========================================

    # Normalize combined anomaly score to 0–100

    min_score = risk_df[
        "combined_anomaly_score"
    ].min()

    max_score = risk_df[
        "combined_anomaly_score"
    ].max()


    if max_score != min_score:

        risk_df["ensemble_ai_score"] = (

            (
                risk_df["combined_anomaly_score"]
                - min_score
            )

            /

            (
                max_score
                - min_score
            )

            * 100

        )

    else:

        risk_df["ensemble_ai_score"] = 0


    # ==========================================
    # BONUS FOR MODEL AGREEMENT
    # ==========================================

    if "both_models_anomaly" in risk_df.columns:

        risk_df.loc[
            risk_df["both_models_anomaly"] == 1,
            "ensemble_ai_score"
        ] += 20


    # Limit score

    risk_df["ensemble_ai_score"] = (

        risk_df["ensemble_ai_score"]
        .clip(0, 100)

    )


    # ==========================================
    # 2. RULE-BASED SCORE
    # ==========================================

    rule_score_map = {

        "LOW": 10,

        "MEDIUM": 50,

        "HIGH": 90

    }


    risk_df["rule_risk_score"] = (

        risk_df["rule_risk_level"]

        .map(rule_score_map)

        .fillna(0)

    )


    # ==========================================
    # 3. FINANCIAL RISK SCORE
    # ==========================================

    risk_df["financial_risk_score"] = 0.0


    # ------------------------------------------
    # Expenditure Ratio
    # ------------------------------------------

    if "expenditure_ratio" in risk_df.columns:

        risk_df.loc[
            risk_df["expenditure_ratio"] > 1.10,
            "financial_risk_score"
        ] += 30


        risk_df.loc[
            (
                risk_df["expenditure_ratio"] > 0.90
            )
            &
            (
                risk_df["expenditure_ratio"] <= 1.10
            ),
            "financial_risk_score"
        ] += 10


    # ------------------------------------------
    # Sanction Difference
    # ------------------------------------------

    if "sanction_difference" in risk_df.columns:

        threshold = (

            risk_df[
                "sanction_difference"
            ]
            .abs()
            .quantile(0.90)

        )


        risk_df.loc[
            risk_df[
                "sanction_difference"
            ].abs() > threshold,
            "financial_risk_score"
        ] += 25


    # ------------------------------------------
    # Vendor Risk
    # ------------------------------------------

    if "unique_vendors" in risk_df.columns:

        risk_df.loc[
            risk_df["unique_vendors"] <= 1,
            "financial_risk_score"
        ] += 15


    # ------------------------------------------
    # Payments Per Vendor
    # ------------------------------------------

    if "payments_per_vendor" in risk_df.columns:

        vendor_threshold = (

            risk_df[
                "payments_per_vendor"
            ]
            .quantile(0.90)

        )


        risk_df.loc[
            risk_df[
                "payments_per_vendor"
            ] > vendor_threshold,
            "financial_risk_score"
        ] += 20


    # Limit financial score

    risk_df["financial_risk_score"] = (

        risk_df["financial_risk_score"]
        .clip(0, 100)

    )


    # ==========================================
    # 4. FINAL HYBRID AI SCORE
    # ==========================================

    risk_df["final_ai_risk_score"] = (

        risk_df["ensemble_ai_score"] * 0.50

        +

        risk_df["rule_risk_score"] * 0.30

        +

        risk_df["financial_risk_score"] * 0.20

    )


    risk_df["final_ai_risk_score"] = (

        risk_df["final_ai_risk_score"]
        .round(2)

    )


    # ==========================================
    # 5. FINAL RISK LEVEL
    # ==========================================

    def get_final_risk_level(score):

        if score >= 75:

            return "CRITICAL"

        elif score >= 55:

            return "HIGH"

        elif score >= 30:

            return "MEDIUM"

        else:

            return "LOW"


    risk_df["final_ai_risk_level"] = (

        risk_df["final_ai_risk_score"]

        .apply(get_final_risk_level)

    )


    # ==========================================
    # RETURN RESULTS
    # ==========================================

    return risk_df