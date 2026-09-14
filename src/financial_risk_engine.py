import pandas as pd
import numpy as np


# ============================================================
# FINANCIAL RISK ENGINE
# ============================================================

def calculate_financial_risk(df):

    # ========================================================
    # CREATE COPY
    # ========================================================

    df = df.copy()


    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    financial_columns = [

        "recommended_amount",
        "sanction_amount",
        "total_expenditure",
        "payment_count",
        "unique_vendors"

    ]


    # ========================================================
    # CHECK COLUMNS
    # ========================================================

    missing_columns = [

        col for col in financial_columns

        if col not in df.columns

    ]

    if missing_columns:

        raise ValueError(

            f"Missing required columns: {missing_columns}"

        )


    # ========================================================
    # CONVERT TO NUMERIC
    # ========================================================

    for col in financial_columns:

        df[col] = pd.to_numeric(

            df[col],

            errors="coerce"

        )

        df[col] = df[col].replace(

            [np.inf, -np.inf],

            np.nan

        )


    # ========================================================
    # HANDLE MISSING VALUES
    # ========================================================

    amount_columns = [

        "recommended_amount",
        "sanction_amount",
        "total_expenditure"

    ]

    count_columns = [

        "payment_count",
        "unique_vendors"

    ]


    for col in amount_columns:

        df[col] = df[col].fillna(0)


    for col in count_columns:

        df[col] = df[col].fillna(0)


    # ========================================================
    # CREATE FINANCIAL RATIOS
    # ========================================================

    # --------------------------------------------------------
    # EXPENDITURE VS SANCTION
    # --------------------------------------------------------

    df["expenditure_vs_sanction"] = np.where(

        df["sanction_amount"] > 0,

        df["total_expenditure"]
        /
        df["sanction_amount"],

        np.nan

    )


    # --------------------------------------------------------
    # EXPENDITURE VS RECOMMENDED
    # --------------------------------------------------------

    df["expenditure_vs_recommended"] = np.where(

        df["recommended_amount"] > 0,

        df["total_expenditure"]
        /
        df["recommended_amount"],

        np.nan

    )


    # --------------------------------------------------------
    # PAYMENTS PER VENDOR
    # --------------------------------------------------------

    df["payments_per_vendor"] = np.where(

        df["unique_vendors"] > 0,

        df["payment_count"]
        /
        df["unique_vendors"],

        0

    )


    # ========================================================
    # DYNAMIC THRESHOLDS
    # ========================================================

    print("\n" + "=" * 60)

    print("DYNAMIC FINANCIAL RISK THRESHOLDS")

    print("=" * 60)


    # --------------------------------------------------------
    # EXPENDITURE THRESHOLDS
    # --------------------------------------------------------

    high_expenditure = max(

        df["total_expenditure"].quantile(0.95),

        500000

    )


    extreme_expenditure = max(

        df["total_expenditure"].quantile(0.99),

        high_expenditure * 1.5

    )


    # --------------------------------------------------------
    # PAYMENT COUNT THRESHOLDS
    # IMPORTANT:
    # Prevent threshold from becoming 1
    # --------------------------------------------------------

    high_payment_count = max(

        df["payment_count"].quantile(0.95),

        5

    )


    extreme_payment_count = max(

        df["payment_count"].quantile(0.99),

        10

    )


    # --------------------------------------------------------
    # PAYMENTS PER VENDOR THRESHOLDS
    # --------------------------------------------------------

    high_ppv = max(

        df["payments_per_vendor"].quantile(0.95),

        3

    )


    extreme_ppv = max(

        df["payments_per_vendor"].quantile(0.99),

        6

    )


    # --------------------------------------------------------
    # PRINT THRESHOLDS
    # --------------------------------------------------------

    print(

        "High expenditure threshold:",

        round(high_expenditure, 2)

    )


    print(

        "Extreme expenditure threshold:",

        round(extreme_expenditure, 2)

    )


    print(

        "High payment count threshold:",

        round(high_payment_count, 2)

    )


    print(

        "Extreme payment count threshold:",

        round(extreme_payment_count, 2)

    )


    print(

        "High payments per vendor threshold:",

        round(high_ppv, 2)

    )


    print(

        "Extreme payments per vendor threshold:",

        round(extreme_ppv, 2)

    )


    # ========================================================
    # CALCULATE FINANCIAL RISK
    # ========================================================

    def calculate_row_financial_risk(row):

        score = 0

        risk_factors = []


        # ====================================================
        # RULE 1
        # EXPENDITURE EXCEEDS SANCTION
        # ====================================================

        ratio = row["expenditure_vs_sanction"]


        if pd.notna(ratio):

            if ratio >= 1.50:

                score += 40

                risk_factors.append(

                    "Expenditure exceeds sanctioned amount by more than 50%"

                )


            elif ratio >= 1.20:

                score += 25

                risk_factors.append(

                    "Expenditure exceeds sanctioned amount by more than 20%"

                )


            elif ratio >= 1.05:

                score += 12

                risk_factors.append(

                    "Expenditure exceeds sanctioned amount by more than 5%"

                )


        # ====================================================
        # RULE 2
        # EXPENDITURE EXCEEDS RECOMMENDED AMOUNT
        # ====================================================

        ratio = row["expenditure_vs_recommended"]


        if pd.notna(ratio):

            if ratio >= 1.50:

                score += 25

                risk_factors.append(

                    "Expenditure is significantly above recommended amount"

                )


            elif ratio >= 1.20:

                score += 15

                risk_factors.append(

                    "Expenditure exceeds recommended amount by more than 20%"

                )


        # ====================================================
        # RULE 3
        # UNUSUALLY HIGH EXPENDITURE
        # ====================================================

        expenditure = row["total_expenditure"]


        if expenditure >= extreme_expenditure:

            score += 20

            risk_factors.append(

                "Project expenditure is among the highest 1%"

            )


        elif expenditure >= high_expenditure:

            score += 10

            risk_factors.append(

                "Project expenditure is unusually high"

            )


        # ====================================================
        # RULE 4
        # HIGH PAYMENT ACTIVITY
        # ====================================================

        payment_count = row["payment_count"]


        if payment_count >= extreme_payment_count:

            score += 12

            risk_factors.append(

                "Extremely high number of payment transactions"

            )


        elif payment_count >= high_payment_count:

            score += 6

            risk_factors.append(

                "Unusually high number of payment transactions"

            )


        # ====================================================
        # RULE 5
        # PAYMENT CONCENTRATION
        #
        # Only meaningful if project has enough payments
        # ====================================================

        ppv = row["payments_per_vendor"]


        if payment_count >= 5:

            if ppv >= extreme_ppv:

                score += 15

                risk_factors.append(

                    "Extremely high payment concentration among vendors"

                )


            elif ppv >= high_ppv:

                score += 8

                risk_factors.append(

                    "High payment concentration among vendors"

                )


        # ====================================================
        # RULE 6
        # MANY PAYMENTS WITH VERY FEW VENDORS
        # ====================================================

        if (

            payment_count >= 10

            and

            row["unique_vendors"] <= 2

        ):

            score += 15

            risk_factors.append(

                "Large number of payments concentrated among very few vendors"

            )


        # ====================================================
        # RULE 7
        # EXTREME SINGLE-VENDOR CONCENTRATION
        # ====================================================

        if (

            payment_count >= 15

            and

            row["unique_vendors"] == 1

        ):

            score += 15

            risk_factors.append(

                "Very high number of payments directed to a single vendor"

            )


        # ====================================================
        # COMBINATION BONUS
        # ====================================================

        original_factor_count = len(risk_factors)


        if original_factor_count >= 3:

            score += 10

            risk_factors.append(

                "Multiple independent financial risk indicators detected"

            )


        # ====================================================
        # FINAL SCORE
        # ====================================================

        score = min(score, 100)


        return pd.Series({

            "financial_risk_score": score,

            "financial_risk_factors": risk_factors,

            "financial_risk_factor_count": len(risk_factors)

        })


    # ========================================================
    # APPLY ENGINE
    # ========================================================

    financial_results = df.apply(

        calculate_row_financial_risk,

        axis=1

    )


    # ========================================================
    # ADD RESULTS
    # ========================================================

    df[

        [

            "financial_risk_score",

            "financial_risk_factors",

            "financial_risk_factor_count"

        ]

    ] = financial_results


    # ========================================================
    # RISK LEVEL CLASSIFICATION
    # ========================================================

    def get_financial_risk_level(score):

        if score >= 60:

            return "CRITICAL"


        elif score >= 40:

            return "HIGH"


        elif score >= 15:

            return "MEDIUM"


        else:

            return "LOW"


    df["financial_risk_level"] = df[

        "financial_risk_score"

    ].apply(

        get_financial_risk_level

    )


    return df
