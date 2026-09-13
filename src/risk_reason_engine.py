import pandas as pd


def generate_risk_reasons(df):

    risk_df = df.copy()

    all_reasons = []

    for _, row in risk_df.iterrows():

        reasons = []

        # --------------------------------------------------
        # 1. AI ANOMALY DETECTION
        # --------------------------------------------------

        if row.get("is_anomaly", 0) == 1:
            reasons.append(
                "Unusual financial pattern detected by the AI model"
            )

        # --------------------------------------------------
        # 2. HIGH PAYMENT ACTIVITY
        # --------------------------------------------------

        payment_count = row.get("payment_count")

        if pd.notna(payment_count) and payment_count >= 5:
            reasons.append(
                f"High number of payments detected ({int(payment_count)} payments)"
            )

        # --------------------------------------------------
        # 3. HIGH PAYMENTS PER VENDOR
        # --------------------------------------------------

        payments_per_vendor = row.get("payments_per_vendor")

        if (
            pd.notna(payments_per_vendor)
            and payments_per_vendor >= 3
        ):
            reasons.append(
                "High number of payments per vendor"
            )

        # --------------------------------------------------
        # 4. UNUSUAL EXPENDITURE RATIO
        # --------------------------------------------------

        expenditure_ratio = row.get("expenditure_ratio")

        if pd.notna(expenditure_ratio):

            if expenditure_ratio < 0.5:
                reasons.append(
                    "Expenditure is significantly lower than the sanctioned amount"
                )

            elif expenditure_ratio > 0.95:
                reasons.append(
                    "Expenditure is very close to the sanctioned amount"
                )

        # --------------------------------------------------
        # 5. MULTIPLE VENDORS
        # --------------------------------------------------

        unique_vendors = row.get("unique_vendors")

        if (
            pd.notna(unique_vendors)
            and unique_vendors >= 5
        ):
            reasons.append(
                f"Large number of vendors involved ({int(unique_vendors)} vendors)"
            )

        # --------------------------------------------------
        # DEFAULT REASON
        # --------------------------------------------------

        if len(reasons) == 0:
            reasons.append(
                "No significant risk indicators detected"
            )

        all_reasons.append(
            " | ".join(reasons)
        )

    risk_df["risk_reasons"] = all_reasons

    return risk_df