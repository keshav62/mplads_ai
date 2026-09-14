import pandas as pd
import numpy as np


# ==========================================
# CALCULATE DYNAMIC THRESHOLDS
# ==========================================

def calculate_thresholds(df):

    thresholds = {}

    # ------------------------------------------
    # PAYMENT COUNT
    # Top 5% projects
    # ------------------------------------------

    if "payment_count" in df.columns:

        thresholds["payment_count"] = (
            df["payment_count"]
            .dropna()
            .quantile(0.95)
        )


    # ------------------------------------------
    # PAYMENTS PER VENDOR
    # ------------------------------------------

    if "payments_per_vendor" in df.columns:

        thresholds["payments_per_vendor"] = (
            df["payments_per_vendor"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .quantile(0.95)
        )


    # ------------------------------------------
    # SANCTION DELAY
    # ------------------------------------------

    if "sanction_delay_days" in df.columns:

        thresholds["sanction_delay_days"] = (
            df["sanction_delay_days"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .quantile(0.95)
        )


    # ------------------------------------------
    # COMPLETION DURATION
    # ------------------------------------------

    if "completion_duration_days" in df.columns:

        thresholds["completion_duration_days"] = (
            df["completion_duration_days"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .quantile(0.95)
        )


    # ------------------------------------------
    # EXPENDITURE RATIO
    # ------------------------------------------

    if "expenditure_ratio" in df.columns:

        thresholds["expenditure_ratio"] = (
            df["expenditure_ratio"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .quantile(0.95)
        )


    return thresholds


# ==========================================
# ANALYZE INDIVIDUAL PROJECT
# ==========================================

def analyze_project_risk(row, thresholds):

    risk_factors = []

    recommended_actions = []

    risk_points = 0


    # ==========================================
    # RULE 1: BOTH ML MODELS DETECTED ANOMALY
    # ==========================================

    if row.get("both_models_anomaly", 0) == 1:

        risk_factors.append(
            "Both anomaly detection models identified this project as unusual."
        )

        recommended_actions.append(
            "Conduct an immediate detailed audit of the project."
        )

        risk_points += 30


    # ==========================================
    # RULE 2: HIGH ENSEMBLE ANOMALY SCORE
    # ==========================================

    combined_score = row.get(
        "combined_anomaly_score",
        0
    )

    if pd.notna(combined_score):

        if combined_score >= 0.55:

            risk_factors.append(
                "Very high combined anomaly score detected."
            )

            recommended_actions.append(
                "Review financial records and project transactions."
            )

            risk_points += 20


        elif combined_score >= 0.45:

            risk_factors.append(
                "Moderately unusual anomaly pattern detected."
            )

            recommended_actions.append(
                "Perform additional monitoring of the project."
            )

            risk_points += 10


    # ==========================================
    # RULE 3: UNUSUALLY HIGH EXPENDITURE RATIO
    # ==========================================

    expenditure_ratio = row.get(
        "expenditure_ratio",
        0
    )

    expenditure_threshold = thresholds.get(
        "expenditure_ratio",
        float("inf")
    )

    if (
        pd.notna(expenditure_ratio)
        and expenditure_ratio > expenditure_threshold
    ):

        risk_factors.append(
            "Expenditure ratio is among the highest values in the dataset."
        )

        recommended_actions.append(
            "Verify expenditure records and budget utilization."
        )

        risk_points += 20


    # ==========================================
    # RULE 4: HIGH PAYMENT COUNT
    # ==========================================

    payment_count = row.get(
        "payment_count",
        0
    )

    payment_threshold = thresholds.get(
        "payment_count",
        float("inf")
    )

    if (
        pd.notna(payment_count)
        and payment_count > payment_threshold
    ):

        risk_factors.append(
            "Payment count is among the highest values in the dataset."
        )

        recommended_actions.append(
            "Review individual payment transactions."
        )

        risk_points += 10


    # ==========================================
    # RULE 5: VENDOR CONCENTRATION
    # ==========================================

    unique_vendors = row.get(
        "unique_vendors",
        0
    )

    if (
        pd.notna(unique_vendors)
        and unique_vendors == 1
        and payment_count > 1
    ):

        risk_factors.append(
            "Multiple payments are concentrated with a single vendor."
        )

        recommended_actions.append(
            "Verify vendor selection and payment distribution."
        )

        risk_points += 15


    # ==========================================
    # RULE 6: HIGH PAYMENTS PER VENDOR
    # ==========================================

    payments_per_vendor = row.get(
        "payments_per_vendor",
        0
    )

    payments_vendor_threshold = thresholds.get(
        "payments_per_vendor",
        float("inf")
    )

    if (
        pd.notna(payments_per_vendor)
        and payments_per_vendor > payments_vendor_threshold
    ):

        risk_factors.append(
            "Payments per vendor are unusually high compared to other projects."
        )

        recommended_actions.append(
            "Review vendor transaction frequency."
        )

        risk_points += 10


    # ==========================================
    # RULE 7: LONG SANCTION DELAY
    # ==========================================

    sanction_delay = row.get(
        "sanction_delay_days",
        0
    )

    sanction_threshold = thresholds.get(
        "sanction_delay_days",
        float("inf")
    )

    if (
        pd.notna(sanction_delay)
        and sanction_delay > sanction_threshold
    ):

        risk_factors.append(
            "Sanction processing delay is among the longest in the dataset."
        )

        recommended_actions.append(
            "Review administrative approval and sanction processing."
        )

        risk_points += 10


    # ==========================================
    # RULE 8: LONG COMPLETION DURATION
    # ==========================================

    completion_duration = row.get(
        "completion_duration_days",
        0
    )

    completion_threshold = thresholds.get(
        "completion_duration_days",
        float("inf")
    )

    if (
        pd.notna(completion_duration)
        and completion_duration > completion_threshold
    ):

        risk_factors.append(
            "Project completion duration is among the longest in the dataset."
        )

        recommended_actions.append(
            "Review project progress and causes of delay."
        )

        risk_points += 10


    # ==========================================
    # DETERMINE RULE-BASED RISK LEVEL
    # ==========================================

    if risk_points >= 50:

        rule_risk_level = "CRITICAL"

    elif risk_points >= 30:

        rule_risk_level = "HIGH"

    elif risk_points >= 15:

        rule_risk_level = "MEDIUM"

    else:

        rule_risk_level = "LOW"


    # ==========================================
    # DEFAULT MESSAGE
    # ==========================================

    if not risk_factors:

        risk_factors.append(
            "No major rule-based risk indicators were detected."
        )

        recommended_actions.append(
            "Continue regular project monitoring."
        )


    # ==========================================
    # RETURN ANALYSIS
    # ==========================================

    return {

        "rule_risk_score": risk_points,

        "rule_risk_level": rule_risk_level,

        "risk_factors": risk_factors,

        "recommended_actions": recommended_actions,

        "risk_factor_count": len(risk_factors)

    }


# ==========================================
# RUN RULE ENGINE ON COMPLETE DATASET
# ==========================================

def run_rule_engine(df):

    # Calculate thresholds from actual dataset

    thresholds = calculate_thresholds(df)


    print("\n" + "=" * 60)

    print("DYNAMIC THRESHOLDS")

    print("=" * 60 + "\n")


    for feature, threshold in thresholds.items():

        print(

            f"{feature}: {threshold:.2f}"

        )


    # ------------------------------------------
    # ANALYZE EACH PROJECT
    # ------------------------------------------

    results = []

    for _, row in df.iterrows():

        analysis = analyze_project_risk(

            row,

            thresholds

        )

        results.append(analysis)


    # ------------------------------------------
    # CREATE RESULTS DATAFRAME
    # ------------------------------------------

    results_df = pd.DataFrame(results)


    # ------------------------------------------
    # MERGE RESULTS
    # ------------------------------------------

    final_df = pd.concat(

        [

            df.reset_index(drop=True),

            results_df.reset_index(drop=True)

        ],

        axis=1

    )


    return final_df