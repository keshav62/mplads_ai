import pandas as pd
import numpy as np


# ============================================================
# FINANCIAL STATISTICAL ANOMALY ENGINE
# ============================================================

def calculate_statistical_anomalies(df):

    # --------------------------------------------------------
    # CREATE COPY
    # --------------------------------------------------------

    df = df.copy()


    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = [

        "recommended_amount",

        "sanction_amount",

        "total_expenditure",

        "payment_count",

        "unique_vendors",

        "payments_per_vendor"

    ]


    available_columns = [

        col for col in required_columns

        if col in df.columns

    ]


    print("\n" + "=" * 60)

    print("IMPROVED STATISTICAL ANOMALY ENGINE")

    print("=" * 60)


    print("\nAvailable financial features:")

    print(available_columns)


    # ========================================================
    # CLEAN NUMERIC DATA
    # ========================================================

    for col in available_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


        df[col] = df[col].replace(
            [np.inf, -np.inf],
            np.nan
        )


        median_value = df[col].median()


        df[col] = df[col].fillna(
            median_value
        )


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

        0

    )


    # --------------------------------------------------------
    # EXPENDITURE VS RECOMMENDED
    # --------------------------------------------------------

    df["expenditure_vs_recommended"] = np.where(

        df["recommended_amount"] > 0,

        df["total_expenditure"]
        /
        df["recommended_amount"],

        0

    )


    # ========================================================
    # INITIALIZE RESULTS
    # ========================================================

    df["statistical_anomaly_score"] = 0

    df["statistical_anomaly_count"] = 0

    df["statistical_anomaly_factors"] = [
        [] for _ in range(len(df))
    ]


    # ========================================================
    # HELPER FUNCTION
    # ========================================================

    def add_anomaly(index, score, factor):

        df.at[
            index,
            "statistical_anomaly_score"
        ] += score


        df.at[
            index,
            "statistical_anomaly_count"
        ] += 1


        df.at[
            index,
            "statistical_anomaly_factors"
        ].append(factor)


    # ========================================================
    # 1. EXPENDITURE ANOMALY
    # LOG TRANSFORM + PERCENTILES
    # ========================================================

    log_expenditure = np.log1p(
        df["total_expenditure"]
    )


    expenditure_p95 = log_expenditure.quantile(0.95)

    expenditure_p99 = log_expenditure.quantile(0.99)

    expenditure_p995 = log_expenditure.quantile(0.995)


    print("\nEXPENDITURE THRESHOLDS")

    print(
        "95th percentile:",
        round(np.expm1(expenditure_p95), 2)
    )

    print(
        "99th percentile:",
        round(np.expm1(expenditure_p99), 2)
    )


    for index in df.index:


        value = log_expenditure.loc[index]


        if value >= expenditure_p995:

            add_anomaly(

                index,

                25,

                "Project expenditure is among the highest 0.5%"

            )


        elif value >= expenditure_p99:

            add_anomaly(

                index,

                18,

                "Project expenditure is among the highest 1%"

            )


        elif value >= expenditure_p95:

            add_anomaly(

                index,

                10,

                "Project expenditure is statistically high"

            )


    # ========================================================
    # 2. PAYMENT COUNT ANOMALY
    # ========================================================

    payment_p95 = df[
        "payment_count"
    ].quantile(0.95)


    payment_p99 = df[
        "payment_count"
    ].quantile(0.99)


    print("\nPAYMENT COUNT THRESHOLDS")

    print(
        "95th percentile:",
        payment_p95
    )

    print(
        "99th percentile:",
        payment_p99
    )


    for index in df.index:


        value = df.at[
            index,
            "payment_count"
        ]


        if value >= payment_p99 and payment_p99 > payment_p95:

            add_anomaly(

                index,

                15,

                "Extremely high number of payment transactions"

            )


        elif value >= payment_p95 and payment_p95 > 1:

            add_anomaly(

                index,

                8,

                "Statistically high number of payment transactions"

            )


    # ========================================================
    # 3. PAYMENTS PER VENDOR ANOMALY
    # ========================================================

    ppv_p95 = df[
        "payments_per_vendor"
    ].quantile(0.95)


    ppv_p99 = df[
        "payments_per_vendor"
    ].quantile(0.99)


    print("\nPAYMENTS PER VENDOR THRESHOLDS")

    print(
        "95th percentile:",
        ppv_p95
    )

    print(
        "99th percentile:",
        ppv_p99
    )


    for index in df.index:


        value = df.at[
            index,
            "payments_per_vendor"
        ]


        if value >= ppv_p99 and ppv_p99 > ppv_p95:

            add_anomaly(

                index,

                15,

                "Extremely high payment concentration per vendor"

            )


        elif value >= ppv_p95 and ppv_p95 > 1:

            add_anomaly(

                index,

                8,

                "High payment concentration per vendor"

            )


    # ========================================================
    # 4. EXPENDITURE VS SANCTION
    # ========================================================

    sanction_ratio = df[
        "expenditure_vs_sanction"
    ]


    ratio_p99 = sanction_ratio.quantile(0.99)


    for index in df.index:


        value = sanction_ratio.loc[index]


        if value > 1.20:

            add_anomaly(

                index,

                20,

                "Expenditure is more than 20% above sanctioned amount"

            )


        elif value > 1.05:

            add_anomaly(

                index,

                12,

                "Expenditure is significantly above sanctioned amount"

            )


        elif value >= ratio_p99 and value > 1:

            add_anomaly(

                index,

                8,

                "Expenditure ratio is statistically unusual"

            )


    # ========================================================
    # 5. EXPENDITURE VS RECOMMENDED
    # ========================================================

    recommended_ratio = df[
        "expenditure_vs_recommended"
    ]


    for index in df.index:


        value = recommended_ratio.loc[index]


        if value > 1.30:

            add_anomaly(

                index,

                18,

                "Expenditure is more than 30% above recommended amount"

            )


        elif value > 1.15:

            add_anomaly(

                index,

                10,

                "Expenditure is significantly above recommended amount"

            )


    # ========================================================
    # MULTIPLE INDEPENDENT ANOMALIES BONUS
    # ========================================================

    multiple_anomalies = (

        df[
            "statistical_anomaly_count"
        ] >= 3

    )


    for index in df[
        multiple_anomalies
    ].index:


        df.at[
            index,
            "statistical_anomaly_score"
        ] += 10


        df.at[
            index,
            "statistical_anomaly_factors"
        ].append(

            "Multiple independent statistical anomalies detected"

        )


    # ========================================================
    # LIMIT SCORE
    # ========================================================

    df[
        "statistical_anomaly_score"
    ] = df[
        "statistical_anomaly_score"
    ].clip(
        upper=100
    )


    # ========================================================
    # ANOMALY LEVEL
    # ========================================================

    def get_anomaly_level(score):

        if score >= 60:

            return "CRITICAL"

        elif score >= 35:

            return "HIGH"

        elif score >= 15:

            return "MEDIUM"

        else:

            return "LOW"


    df[
        "statistical_anomaly_level"
    ] = df[
        "statistical_anomaly_score"
    ].apply(
        get_anomaly_level
    )


    return df