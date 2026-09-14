import pandas as pd


def generate_investigation_report(row):

    explanations = []
    actions = []

    # =====================================================
    # ML ANOMALY ENGINE
    # =====================================================

    if row.get("ensemble_is_anomaly", 0) == 1:

        explanations.append(
            "Machine learning anomaly models detected unusual project behavior compared to similar projects."
        )

        actions.append(
            "Review the project for unusual patterns compared with historical projects."
        )

    # =====================================================
    # RULE-BASED RISK ENGINE
    # =====================================================

    if row.get("rule_risk_score", 0) >= 25:

        explanations.append(
            "Rule-based validation detected one or more predefined risk conditions."
        )

        actions.append(
            "Review the triggered rule-based risk conditions."
        )

    # =====================================================
    # FINANCIAL RISK ENGINE
    # =====================================================

    financial_score = row.get(
        "financial_risk_score",
        0
    )

    if financial_score >= 25:

        explanations.append(
            "Financial analysis detected unusual expenditure, payment, or vendor patterns."
        )

        actions.append(
            "Perform detailed financial verification of expenditure and payment records."
        )

    # =====================================================
    # STATISTICAL ANOMALY ENGINE
    # =====================================================

    statistical_score = row.get(
        "statistical_anomaly_score",
        0
    )

    if statistical_score >= 25:

        explanations.append(
            "Statistical analysis detected values significantly different from the overall project population."
        )

        actions.append(
            "Compare this project with similar projects in the same category and region."
        )

    # =====================================================
    # FINANCIAL RISK FACTORS
    # =====================================================

    financial_factors = row.get(
        "financial_risk_factors",
        ""
    )

    if isinstance(financial_factors, str):

        if financial_factors.strip():

            explanations.append(
                f"Financial indicators: {financial_factors}"
            )

    # =====================================================
    # STATISTICAL ANOMALY FACTORS
    # =====================================================

    statistical_factors = row.get(
        "statistical_anomaly_factors",
        ""
    )

    if isinstance(statistical_factors, str):

        if statistical_factors.strip():

            explanations.append(
                f"Statistical indicators: {statistical_factors}"
            )

    # =====================================================
    # CONSENSUS
    # =====================================================

    active_engines = row.get(
        "active_risk_engines",
        0
    )

    if active_engines >= 4:

        explanations.append(
            "All four independent risk engines detected suspicious signals."
        )

        actions.append(
            "Prioritize this project for immediate investigation."
        )

    elif active_engines == 3:

        explanations.append(
            "Three independent risk engines detected suspicious signals."
        )

        actions.append(
            "Schedule a detailed risk review."
        )

    elif active_engines == 2:

        explanations.append(
            "Two independent risk engines detected suspicious signals."
        )

        actions.append(
            "Conduct a preliminary verification."
        )

    # =====================================================
    # FINAL RISK LEVEL ACTION
    # =====================================================

    risk_level = row.get(
        "final_ai_risk_level",
        "LOW"
    )

    if risk_level == "CRITICAL":

        actions.append(
            "Initiate immediate financial and administrative investigation."
        )

        actions.append(
            "Verify all payments, vendors, approvals, and expenditure records."
        )

    elif risk_level == "HIGH":

        actions.append(
            "Prioritize this project for detailed audit."
        )

    elif risk_level == "MEDIUM":

        actions.append(
            "Monitor the project and perform additional verification."
        )

    else:

        actions.append(
            "Continue normal monitoring."
        )

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    explanations = list(dict.fromkeys(explanations))

    actions = list(dict.fromkeys(actions))

    # =====================================================
    # RETURN REPORT
    # =====================================================

    return {

        "risk_explanation": explanations,

        "recommended_actions": actions,

        "explanation_count": len(explanations),

        "recommended_action_count": len(actions)

    }


def generate_investigation_reports(df):

    print("\nGenerating explainable AI investigation reports...")

    results = df.apply(
        lambda row: pd.Series(
            generate_investigation_report(row)
        ),
        axis=1
    )

    df = df.copy()

    df[
        [
            "risk_explanation",
            "recommended_actions",
            "explanation_count",
            "recommended_action_count"
        ]
    ] = results

    return df