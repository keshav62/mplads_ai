import pandas as pd
import numpy as np


# ============================================================
# UNIFIED FINAL RISK INTELLIGENCE ENGINE
# ============================================================

def calculate_unified_risk(
    ml_df,
    rule_df,
    financial_df,
    statistical_df
):

    print("\n" + "=" * 60)
    print("UNIFIED RISK INTELLIGENCE ENGINE")
    print("=" * 60)

    # ========================================================
    # CREATE COPIES
    # ========================================================

    ml_df = ml_df.copy()
    rule_df = rule_df.copy()
    financial_df = financial_df.copy()
    statistical_df = statistical_df.copy()

    # ========================================================
    # KEEP REQUIRED COLUMNS
    # ========================================================

    # ML ENGINE
    ml_columns = [
        "work_id",
        "combined_anomaly_score",
        "ensemble_risk_level"
    ]

    ml_available = [
        col for col in ml_columns
        if col in ml_df.columns
    ]

    ml_data = ml_df[ml_available].copy()

    # --------------------------------------------------------

    # RULE ENGINE
    rule_columns = [
        "work_id",
        "rule_risk_score",
        "rule_risk_level",
        "risk_factors",
        "recommended_actions"
    ]

    rule_available = [
        col for col in rule_columns
        if col in rule_df.columns
    ]

    rule_data = rule_df[rule_available].copy()

    # --------------------------------------------------------

    # FINANCIAL ENGINE
    financial_columns = [
        "work_id",
        "financial_risk_score",
        "financial_risk_level",
        "financial_risk_factors"
    ]

    financial_available = [
        col for col in financial_columns
        if col in financial_df.columns
    ]

    financial_data = financial_df[
        financial_available
    ].copy()

    # --------------------------------------------------------

    # STATISTICAL ENGINE
    statistical_columns = [
        "work_id",
        "statistical_anomaly_score",
        "statistical_anomaly_level",
        "statistical_anomaly_factors"
    ]

    statistical_available = [
        col for col in statistical_columns
        if col in statistical_df.columns
    ]

    statistical_data = statistical_df[
        statistical_available
    ].copy()

    # ========================================================
    # MERGE DATASETS
    # ========================================================

    print("\nMerging risk engines...")

    df = ml_data.merge(
        rule_data,
        on="work_id",
        how="outer"
    )

    df = df.merge(
        financial_data,
        on="work_id",
        how="outer"
    )

    df = df.merge(
        statistical_data,
        on="work_id",
        how="outer"
    )

    print("Total projects after merge:", len(df))

    # ========================================================
    # CONVERT SCORES TO NUMERIC
    # ========================================================

    score_columns = [
        "combined_anomaly_score",
        "rule_risk_score",
        "financial_risk_score",
        "statistical_anomaly_score"
    ]

    for col in score_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            df[col] = df[col].fillna(0)

    # ========================================================
    # NORMALIZE ML SCORE
    # ========================================================

    if "combined_anomaly_score" in df.columns:

        ml_max = df[
            "combined_anomaly_score"
        ].max()

        if ml_max > 0:

            df["ml_normalized_score"] = (
                df["combined_anomaly_score"]
                / ml_max
            ) * 100

        else:

            df["ml_normalized_score"] = 0

    else:

        df["ml_normalized_score"] = 0

    # ========================================================
    # NORMALIZE OTHER SCORES
    # ========================================================

    for col in [
        "rule_risk_score",
        "financial_risk_score",
        "statistical_anomaly_score"
    ]:

        if col not in df.columns:

            df[col] = 0

        df[col] = df[col].clip(
            lower=0,
            upper=100
        )

    # ========================================================
    # WEIGHTED FINAL RISK SCORE
    # ========================================================

    print("\nRISK WEIGHTS")

    print("ML Anomaly Score:         35%")
    print("Rule-Based Risk Score:    25%")
    print("Financial Risk Score:     20%")
    print("Statistical Anomaly:      20%")

    df["final_ai_risk_score"] = (

        df["ml_normalized_score"] * 0.35

        +

        df["rule_risk_score"] * 0.25

        +

        df["financial_risk_score"] * 0.20

        +

        df["statistical_anomaly_score"] * 0.20

    )

    df["final_ai_risk_score"] = (
        df["final_ai_risk_score"]
        .clip(0, 100)
        .round(2)
    )

    # ========================================================
    # RISK LEVEL CLASSIFICATION
    # ========================================================

    def get_final_risk_level(score):

        if score >= 75:
            return "CRITICAL"

        elif score >= 50:
            return "HIGH"

        elif score >= 25:
            return "MEDIUM"

        else:
            return "LOW"

    df["final_ai_risk_level"] = df[
        "final_ai_risk_score"
    ].apply(
        get_final_risk_level
    )

    # ========================================================
    # CRITICAL SIGNAL OVERRIDE
    # ========================================================

    def apply_risk_override(row):

        current_level = row[
            "final_ai_risk_level"
        ]

        critical_signals = 0

        # ML Critical

        if (
            row.get("ensemble_risk_level")
            == "CRITICAL"
        ):

            critical_signals += 1

        # Rule Critical

        if (
            row.get("rule_risk_level")
            == "CRITICAL"
        ):

            critical_signals += 1

        # Financial Critical

        if (
            row.get("financial_risk_level")
            == "CRITICAL"
        ):

            critical_signals += 1

        # Statistical Critical

        if (
            row.get("statistical_anomaly_level")
            == "CRITICAL"
        ):

            critical_signals += 1

        # ----------------------------------------------------

        # Two or more critical signals

        if critical_signals >= 2:

            return "CRITICAL"

        # One critical signal should at least be HIGH

        elif critical_signals == 1:

            levels = [
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL"
            ]

            if levels.index(current_level) < levels.index("HIGH"):

                return "HIGH"

        return current_level

    df["final_ai_risk_level"] = df.apply(
        apply_risk_override,
        axis=1
    )

    # ========================================================
    # COUNT ACTIVE RISK ENGINES
    # ========================================================

    def count_active_engines(row):

        active = 0

        if row.get(
            "ml_normalized_score",
            0
        ) > 0:

            active += 1

        if row.get(
            "rule_risk_score",
            0
        ) > 0:

            active += 1

        if row.get(
            "financial_risk_score",
            0
        ) > 0:

            active += 1

        if row.get(
            "statistical_anomaly_score",
            0
        ) > 0:

            active += 1

        return active

    df["active_risk_engines"] = df.apply(
        count_active_engines,
        axis=1
    )

    # ========================================================
    # GENERATE FINAL RISK EXPLANATION
    # ========================================================

    def generate_final_risk_factors(row):

        factors = []

        # ML

        if row.get(
            "ml_normalized_score",
            0
        ) > 0:

            if row.get(
                "ensemble_risk_level"
            ) in ["HIGH", "CRITICAL"]:

                factors.append(
                    "Machine learning anomaly models detected unusual project behavior"
                )

        # RULE

        if row.get(
            "rule_risk_score",
            0
        ) > 0:

            factors.append(
                "Rule-based risk indicators were detected"
            )

        # FINANCIAL

        if row.get(
            "financial_risk_score",
            0
        ) > 0:

            factors.append(
                "Financial risk indicators were detected"
            )

        # STATISTICAL

        if row.get(
            "statistical_anomaly_score",
            0
        ) > 0:

            factors.append(
                "Statistical financial anomalies were detected"
            )

        # MULTIPLE ENGINES

        if row[
            "active_risk_engines"
        ] >= 3:

            factors.append(
                "Multiple independent AI risk engines detected suspicious patterns"
            )

        return factors

    df["final_ai_risk_factors"] = df.apply(
        generate_final_risk_factors,
        axis=1
    )

    # ========================================================
    # CONFIDENCE SCORE
    # ========================================================

    def calculate_confidence(row):

        active = row[
            "active_risk_engines"
        ]

        if active >= 4:
            return "VERY HIGH"

        elif active == 3:
            return "HIGH"

        elif active == 2:
            return "MEDIUM"

        elif active == 1:
            return "LOW"

        return "VERY LOW"

    df["risk_detection_confidence"] = df.apply(
        calculate_confidence,
        axis=1
    )

    print("\nUnified risk calculation completed.")

    return df