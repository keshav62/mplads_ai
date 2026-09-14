# ============================================================
# INVESTIGATION & REPORTING LAYER
# Government Project Risk Intelligence System
#
# Input:
#   final_unified_risk_analysis_v2.csv
#
# Outputs:
#   investigation_queue.csv
#   investigation_summary.csv
#   grounded_llm_investigation_reports_v4.csv
#   investigation_reports/*.md
#
# ============================================================

import os
import json
import ast
import re
import time

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = (
    "data/processed/"
    "final_unified_risk_analysis_v2.csv"
)

QUEUE_OUTPUT_PATH = (
    "data/processed/"
    "investigation_queue.csv"
)

SUMMARY_OUTPUT_PATH = (
    "data/processed/"
    "investigation_summary.csv"
)

LLM_OUTPUT_PATH = (
    "data/processed/"
    "grounded_llm_investigation_reports_v4.csv"
)

REPORT_DIRECTORY = (
    "data/processed/"
    "investigation_reports"
)


# ------------------------------------------------------------
# Number of LLM reports to generate
#
# IMPORTANT:
# The investigation queue contains ALL HIGH + CRITICAL
# projects.
#
# LLM reports can be limited to the highest priority
# projects to control API usage.
# ------------------------------------------------------------

MAX_LLM_REPORTS = 20


# ------------------------------------------------------------
# Groq configuration
# ------------------------------------------------------------

GROQ_MODEL = "openai/gpt-oss-20b"

TEMPERATURE = 0


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


# ============================================================
# INITIALIZE LLM
# ============================================================

llm = None


if GROQ_API_KEY:

    llm = ChatGroq(

        model=GROQ_MODEL,

        api_key=GROQ_API_KEY,

        temperature=TEMPERATURE

    )

else:

    print(
        "\nWARNING:"
        "\nGROQ_API_KEY was not found."
        "\nLLM reports will be skipped."
        "\nDeterministic investigation queue will still be generated.\n"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_missing(value):

    if value is None:

        return True


    if isinstance(
        value,
        np.ndarray
    ):

        return False


    if isinstance(
        value,
        (list, dict)
    ):

        return False


    try:

        result = pd.isna(value)


        if isinstance(
            result,
            (bool, np.bool_)
        ):

            return bool(result)


        return False

    except Exception:

        return False


# ============================================================

def safe_value(
    value,
    default="Not available in the provided evidence."
):

    if is_missing(value):

        return default


    if isinstance(
        value,
        np.generic
    ):

        return value.item()


    if isinstance(
        value,
        np.ndarray
    ):

        return value.tolist()


    return value


# ============================================================

def parse_list(value):

    if is_missing(value):

        return []


    if isinstance(
        value,
        list
    ):

        return value


    if isinstance(
        value,
        tuple
    ):

        return list(value)


    if isinstance(
        value,
        np.ndarray
    ):

        return value.tolist()


    if isinstance(
        value,
        str
    ):

        value = value.strip()


        if not value:

            return []


        try:

            parsed = ast.literal_eval(
                value
            )


            if isinstance(
                parsed,
                list
            ):

                return parsed


            if isinstance(
                parsed,
                tuple
            ):

                return list(parsed)

        except Exception:

            pass


        return [value]


    return [safe_value(value)]


# ============================================================

def get_column(
    row,
    column_name,
    default="Not available in the provided evidence."
):

    if column_name not in row.index:

        return default


    value = row[column_name]


    return safe_value(
        value,
        default
    )


# ============================================================

def clean_filename(work_id):

    filename = str(
        work_id
    )

    filename = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        filename
    )

    return filename


# ============================================================

def safe_float(
    value,
    decimals=2
):

    if is_missing(value):

        return None


    try:

        return round(
            float(value),
            decimals
        )

    except Exception:

        return None


# ============================================================
# ENGINE PRIORITY
# ============================================================

def calculate_investigation_priority(row):

    score = safe_float(
        row.get(
            "final_ai_risk_score"
        ),
        2
    )


    active_engines = int(

        safe_float(

            row.get(
                "active_risk_engines",
                0
            ),

            0

        ) or 0

    )


    level = str(

        row.get(
            "final_ai_risk_level",
            ""
        )

    ).upper()


    priority = 0


    # --------------------------------------------------------
    # Final risk level
    # --------------------------------------------------------

    if level == "CRITICAL":

        priority += 50

    elif level == "HIGH":

        priority += 30


    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    if score is not None:

        priority += (
            score * 0.40
        )


    # --------------------------------------------------------
    # Multi-engine evidence
    # --------------------------------------------------------

    if active_engines >= 4:

        priority += 25

    elif active_engines == 3:

        priority += 18

    elif active_engines == 2:

        priority += 10

    elif active_engines == 1:

        priority += 3


    return round(
    min(priority, 100),
    2
    )


# ============================================================
# INVESTIGATION PRIORITY CATEGORY
# ============================================================

def get_priority_category(
    row
):

    level = str(

        row.get(
            "final_ai_risk_level",
            ""
        )

    ).upper()


    engines = int(

        safe_float(

            row.get(
                "active_risk_engines",
                0
            ),

            0

        ) or 0

    )


    if level == "CRITICAL":

        if engines >= 3:

            return "P1 - IMMEDIATE REVIEW"

        return "P1 - PRIORITY REVIEW"


    if level == "HIGH":

        if engines >= 3:

            return "P2 - HIGH PRIORITY"

        return "P3 - STANDARD REVIEW"


    return "MONITOR"


# ============================================================
# BUILD VERIFIED PROJECT EVIDENCE
# ============================================================

def build_project_evidence(
    project
):

    evidence = {

        # ====================================================
        # PROJECT INFORMATION
        # ====================================================

        "project_information": {

            "work_id":
                get_column(
                    project,
                    "work_id"
                ),

            "state":
                get_column(
                    project,
                    "state"
                ),

            "constituency":
                get_column(
                    project,
                    "constituency"
                ),

            "work_category":
                get_column(
                    project,
                    "work_category"
                ),

            "work_description":
                get_column(
                    project,
                    "work_description"
                ),

            "work_status":
                get_column(
                    project,
                    "work_status"
                ),

            "financial_year":
                get_column(
                    project,
                    "financial_year"
                )

        },


        # ====================================================
        # FINANCIAL INFORMATION
        # ====================================================

        "financial_information": {

            "recommended_amount":
                get_column(
                    project,
                    "recommended_amount"
                ),

            "sanction_amount":
                get_column(
                    project,
                    "sanction_amount"
                ),

            "total_expenditure":
                get_column(
                    project,
                    "total_expenditure"
                ),

            "completed_amount_disbursed":
                get_column(
                    project,
                    "completed_amount_disbursed"
                ),

            "expenditure_ratio":
                get_column(
                    project,
                    "expenditure_ratio"
                ),

            "sanction_ratio":
                get_column(
                    project,
                    "sanction_ratio"
                ),

            "expenditure_difference":
                get_column(
                    project,
                    "expenditure_difference"
                ),

            "sanction_difference":
                get_column(
                    project,
                    "sanction_difference"
                )

        },


        # ====================================================
        # PAYMENT INFORMATION
        # ====================================================

        "payment_information": {

            "payment_count":
                get_column(
                    project,
                    "payment_count"
                ),

            "unique_vendors":
                get_column(
                    project,
                    "unique_vendors"
                ),

            "payments_per_vendor":
                get_column(
                    project,
                    "payments_per_vendor"
                )

        },


        # ====================================================
        # LIFECYCLE INFORMATION
        # ====================================================

        "project_lifecycle": {

            "recommended_date":
                get_column(
                    project,
                    "recommended_date"
                ),

            "sanction_date":
                get_column(
                    project,
                    "sanction_date"
                ),

            "completion_date":
                get_column(
                    project,
                    "completion_date"
                ),

            "sanction_delay_days":
                get_column(
                    project,
                    "sanction_delay_days"
                ),

            "completion_duration_days":
                get_column(
                    project,
                    "completion_duration_days"
                )

        },


        # ====================================================
        # UNIFIED AI RISK ENGINE
        # ====================================================

        "unified_ai_risk": {

            "final_ai_risk_score":
                get_column(
                    project,
                    "final_ai_risk_score"
                ),

            "final_ai_risk_level":
                get_column(
                    project,
                    "final_ai_risk_level"
                ),

            "risk_detection_confidence":
                get_column(
                    project,
                    "risk_detection_confidence"
                ),

            "active_risk_engines":
                get_column(
                    project,
                    "active_risk_engines"
                ),

            "available_risk_engines":
                get_column(
                    project,
                    "available_risk_engines"
                ),

            "consensus_bonus":
                get_column(
                    project,
                    "consensus_bonus"
                ),

            "primary_risk_source":
                get_column(
                    project,
                    "primary_risk_source"
                ),

            "detecting_engines":
                get_column(
                    project,
                    "detecting_engines"
                )

        },


        # ====================================================
        # ML ENGINE
        # ====================================================

        "machine_learning_engine": {

            "ml_anomaly_score":
                get_column(
                    project,
                    "ml_anomaly_score"
                ),

            "ml_detected":
                get_column(
                    project,
                    "ml_detected"
                ),

            "ml_model_agreement":
                get_column(
                    project,
                    "model_agreement"
                ),

            "ensemble_risk_level":
                get_column(
                    project,
                    "ensemble_risk_level"
                ),

            "ensemble_is_anomaly":
                get_column(
                    project,
                    "ensemble_is_anomaly"
                )

        },


        # ====================================================
        # RULE ENGINE
        # ====================================================

        "rule_based_engine": {

            "rule_risk_score":
                get_column(
                    project,
                    "rule_risk_score"
                ),

            "rule_detected":
                get_column(
                    project,
                    "rule_detected"
                ),

            "risk_factors":
                parse_list(
                    get_column(
                        project,
                        "risk_factors",
                        []
                    )
                )

        },


        # ====================================================
        # FINANCIAL ENGINE
        # ====================================================

        "financial_risk_engine": {

            "financial_risk_score":
                get_column(
                    project,
                    "financial_risk_score"
                ),

            "financial_detected":
                get_column(
                    project,
                    "financial_detected"
                ),

            "financial_risk_factors":
                parse_list(
                    get_column(
                        project,
                        "financial_risk_factors",
                        []
                    )
                )

        },


        # ====================================================
        # STATISTICAL ENGINE
        # ====================================================

        "statistical_anomaly_engine": {

            "statistical_anomaly_score":
                get_column(
                    project,
                    "statistical_anomaly_score"
                ),

            "statistical_detected":
                get_column(
                    project,
                    "statistical_detected"
                ),

            "statistical_anomaly_level":
                get_column(
                    project,
                    "statistical_anomaly_level"
                ),

            "statistical_anomaly_factors":
                parse_list(
                    get_column(
                        project,
                        "statistical_anomaly_factors",
                        []
                    )
                )

        },


        # ====================================================
        # COMBINED FACTORS
        # ====================================================

        "combined_risk_information": {

            "combined_risk_factors":
                parse_list(
                    get_column(
                        project,
                        "combined_risk_factors",
                        []
                    )
                ),

            "combined_risk_factor_count":
                get_column(
                    project,
                    "combined_risk_factor_count"
                )

        }

    }


    return evidence


# ============================================================
# BUILD DETERMINISTIC INVESTIGATION SUMMARY
# ============================================================

def build_investigation_summary(
    project
):

    work_id = get_column(
        project,
        "work_id"
    )


    risk_level = get_column(
        project,
        "final_ai_risk_level"
    )


    risk_score = get_column(
        project,
        "final_ai_risk_score"
    )


    engines = get_column(
        project,
        "detecting_engines"
    )


    priority = get_column(
        project,
        "investigation_priority_score"
    )


    priority_category = get_column(
        project,
        "investigation_priority_category"
    )


    observations = []


    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    if not is_missing(
        project.get(
            "final_ai_risk_score"
        )
    ):

        observations.append(

            "Final AI risk score: "
            + str(
                project[
                    "final_ai_risk_score"
                ]
            )

        )


    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    observations.append(

        "Final AI risk level: "
        + str(
            risk_level
        )

    )


    # --------------------------------------------------------
    # Multiple engine detection
    # --------------------------------------------------------

    active_engines = int(

        safe_float(

            project.get(
                "active_risk_engines",
                0
            ),

            0

        ) or 0

    )


    if active_engines >= 2:

        observations.append(

            f"{active_engines} independent "
            "risk engines detected significant "
            "risk signals."

        )


    elif active_engines == 1:

        observations.append(

            "One risk engine detected "
            "a significant risk signal."

        )


    # --------------------------------------------------------
    # Financial indicators
    # --------------------------------------------------------

    expenditure_ratio = project.get(
        "expenditure_ratio"
    )


    if not is_missing(
        expenditure_ratio
    ):

        observations.append(

            "Recorded expenditure ratio: "
            + str(
                expenditure_ratio
            )

        )


    # --------------------------------------------------------
    # Payment indicators
    # --------------------------------------------------------

    payment_count = project.get(
        "payment_count"
    )


    if not is_missing(
        payment_count
    ):

        observations.append(

            "Payment count: "
            + str(
                payment_count
            )

        )


    unique_vendors = project.get(
        "unique_vendors"
    )


    if not is_missing(
        unique_vendors
    ):

        observations.append(

            "Unique vendors: "
            + str(
                unique_vendors
            )

        )


    return {

        "work_id":
            work_id,

        "risk_level":
            risk_level,

        "risk_score":
            risk_score,

        "priority_score":
            priority,

        "priority_category":
            priority_category,

        "active_risk_engines":
            active_engines,

        "detecting_engines":
            engines,

        "verified_observations":
            " | ".join(
                observations
            )

    }


# ============================================================
# GROUNDED LLM PROMPT
# ============================================================

def create_investigation_prompt(
    evidence
):

    evidence_json = json.dumps(

        evidence,

        indent=2,

        default=str

    )


    prompt = f"""
You are an AI Investigation Copilot for a
Government Project Risk Intelligence System.

Your task is to prepare a professional,
evidence-grounded preliminary investigation report.

You MUST use ONLY the verified project evidence
provided below.

============================================================
STRICT GROUNDING RULES
============================================================

1. Use ONLY the provided evidence.

2. NEVER invent:
   - vendors
   - transactions
   - financial amounts
   - dates
   - percentages
   - beneficiaries
   - contractors
   - allegations
   - missing information

3. NEVER state or imply that fraud, corruption,
   financial misconduct, criminal activity, or
   misuse of funds has occurred.

4. Risk scores and anomaly detections are indicators
   for investigation. They are NOT proof of wrongdoing.

5. Clearly distinguish:
   - Verified Observation
   - Risk Indicator
   - AI/ML Detection
   - Investigation Recommendation

6. If information is unavailable, write:
   "Not available in the provided evidence."

7. Do not change numerical values.

8. When multiple engines detect risk, explain that
   multiple analytical signals agree, but this does
   NOT establish wrongdoing.

9. The report should help a human government auditor
   decide what evidence should be verified next.

10. Do not make legal conclusions.

11. Do not make accusations.

12. Be precise and conservative.

============================================================
VERIFIED PROJECT EVIDENCE
============================================================

{evidence_json}

============================================================
REPORT FORMAT
============================================================

Generate EXACTLY the following sections:

# INVESTIGATION REPORT

# 1. EXECUTIVE SUMMARY

Explain briefly why this project has been prioritized.

Include:
- final risk level
- final AI risk score
- number of detecting engines
- primary risk source
- why human review is recommended

# 2. PROJECT OVERVIEW

Summarize:
- work ID
- state
- constituency
- category
- description
- status
- financial year

# 3. FINANCIAL OBSERVATIONS

Discuss only the available:
- recommended amount
- sanctioned amount
- expenditure
- expenditure ratio
- sanction ratio
- differences

Do not infer wrongdoing.

# 4. PAYMENT OBSERVATIONS

Discuss:
- payment count
- unique vendors
- payments per vendor

Only state what the evidence shows.

# 5. PROJECT LIFECYCLE OBSERVATIONS

Discuss available:
- recommendation date
- sanction date
- completion date
- sanction delay
- completion duration

# 6. AI/ML RISK SIGNALS

Explain:

ML detection:
Rule-based detection:
Financial detection:
Statistical detection:
Active engine count:
Model agreement:
Primary risk source:

Explain what the signals indicate.

Do NOT call them proof of wrongdoing.

# 7. RISK FACTORS

List the available risk factors.

Separate:
- rule-based factors
- financial factors
- statistical factors
- combined factors

If no factors are available, say:
"Not available in the provided evidence."

# 8. INVESTIGATION PRIORITIES

Provide a numbered list of concrete verification steps.

Examples:

1. Verify sanction records.
2. Verify expenditure records.
3. Reconcile payment records.
4. Verify completion documentation.
5. Verify supporting project records.
6. Compare reported financial values with source records.

Only recommend checks relevant to the evidence.

# 9. REQUIRED EVIDENCE

List documents/data that a human investigator
should obtain or verify.

Examples:
- sanction documentation
- expenditure records
- payment records
- completion records
- project measurement records
- supporting administrative records

Do not claim that these records are missing.
Say they should be verified where relevant.

# 10. INVESTIGATION CONCLUSION

Provide a short conclusion.

Use language such as:

"This project is prioritized for human review because
the available evidence contains multiple risk indicators."

Explicitly state:

"The risk indicators do not establish wrongdoing."

============================================================
END OF REPORT
============================================================
"""

    return prompt


# ============================================================
# GENERATE LLM REPORT
# ============================================================

def generate_grounded_report(
    project
):

    if llm is None:

        return (
            "LLM report unavailable because "
            "GROQ_API_KEY was not configured."
        )


    evidence = build_project_evidence(
        project
    )


    prompt = create_investigation_prompt(
        evidence
    )


    response = llm.invoke(
        [
            (
                "system",
                "You are a careful government "
                "risk investigation copilot. "
                "Never invent evidence."
            ),

            (
                "user",
                prompt
            )

        ]
    )


    return response.content.strip()


# ============================================================
# CREATE MARKDOWN REPORT
# ============================================================

def create_markdown_report(
    project,
    llm_report
):

    work_id = get_column(
        project,
        "work_id"
    )


    risk_score = get_column(
        project,
        "final_ai_risk_score"
    )


    risk_level = get_column(
        project,
        "final_ai_risk_level"
    )


    priority = get_column(
        project,
        "investigation_priority_score"
    )


    priority_category = get_column(
        project,
        "investigation_priority_category"
    )


    engines = get_column(
        project,
        "detecting_engines"
    )


    header = f"""
# Government Project Investigation Report

**Work ID:** {work_id}

**Final AI Risk Level:** {risk_level}

**Final AI Risk Score:** {risk_score}

**Investigation Priority Score:** {priority}

**Investigation Priority:** {priority_category}

**Detecting Engines:** {engines}

---

## Important Notice

This report is an AI-assisted investigation aid.

Risk scores, statistical anomalies, rules, and machine-learning
detections are indicators for human review.

**They do not establish fraud, corruption, misuse of funds,
or any other wrongdoing.**

---

"""


    return (
        header
        +
        llm_report
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "INVESTIGATION & REPORTING LAYER"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # LOAD UNIFIED RISK DATA
    # ========================================================

    print(
        "\nLoading unified risk dataset..."
    )


    if not os.path.exists(
        INPUT_PATH
    ):

        raise FileNotFoundError(

            f"Unified risk dataset not found:\n"
            f"{INPUT_PATH}\n\n"
            "Run train_unified_risk_v2.py first."

        )


    df = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)


    print(
        f"Projects loaded: {len(df)}"
    )


    # ========================================================
    # VALIDATE REQUIRED COLUMNS
    # ========================================================

    required_columns = [

        "work_id",

        "final_ai_risk_score",

        "final_ai_risk_level",

        "active_risk_engines"

    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns

    ]


    if missing_columns:

        raise ValueError(

            "Required columns missing:\n"
            + "\n".join(
                missing_columns
            )

        )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    if not df[
        "work_id"
    ].is_unique:

        raise ValueError(
            "Duplicate work_id values detected."
        )


    # ========================================================
    # SELECT HIGH + CRITICAL
    # ========================================================

    investigation_df = df[

        df[
            "final_ai_risk_level"
        ]

        .astype(str)
        .str.upper()
        .isin(

            [
                "HIGH",
                "CRITICAL"
            ]

        )

    ].copy()


    print(
        "\nInvestigation candidates:"
    )


    print(
        investigation_df[
            "final_ai_risk_level"
        ]
        .value_counts()
    )


    print(
        "\nTotal HIGH + CRITICAL:",
        len(
            investigation_df
        )
    )


    # ========================================================
    # CALCULATE INVESTIGATION PRIORITY
    # ========================================================

    investigation_df[
        "investigation_priority_score"
    ] = (

        investigation_df.apply(

            calculate_investigation_priority,

            axis=1

        )

    )


    investigation_df[
        "investigation_priority_category"
    ] = (

        investigation_df.apply(

            get_priority_category,

            axis=1

        )

    )


    # ========================================================
    # SORT INVESTIGATION QUEUE
    # ========================================================

    investigation_df = (

        investigation_df

        .sort_values(

            [

                "investigation_priority_score",

                "final_ai_risk_score",

                "active_risk_engines"

            ],

            ascending=False

        )

        .reset_index(
            drop=True
        )

    )


    # ========================================================
    # ASSIGN QUEUE RANK
    # ========================================================

    investigation_df[
        "investigation_rank"
    ] = (

        np.arange(
            1,
            len(
                investigation_df
            ) + 1
        )

    )


    # ========================================================
    # CREATE DETERMINISTIC SUMMARIES
    # ========================================================

    summary_rows = []


    for _, row in investigation_df.iterrows():

        summary_rows.append(

            build_investigation_summary(
                row
            )

        )


    summary_df = pd.DataFrame(
        summary_rows
    )


    # ========================================================
    # MERGE SUMMARY
    # ========================================================

    investigation_df = investigation_df.merge(

        summary_df,

        on="work_id",

        how="left",

        suffixes=(
            "",
            "_summary"
        )

    )


    # ========================================================
    # SAVE INVESTIGATION QUEUE
    # ========================================================

    os.makedirs(
        "data/processed",
        exist_ok=True
    )


    investigation_df.to_csv(

        QUEUE_OUTPUT_PATH,

        index=False

    )


    print(
        "\nInvestigation queue saved:"
    )

    print(
        QUEUE_OUTPUT_PATH
    )


    # ========================================================
    # CREATE SUMMARY STATISTICS
    # ========================================================

    summary_rows = []


    total_projects = len(
        investigation_df
    )


    critical_count = int(

        (
            investigation_df[
                "final_ai_risk_level"
            ]
            .astype(str)
            .str.upper()
            == "CRITICAL"
        )

        .sum()

    )


    high_count = int(

        (
            investigation_df[
                "final_ai_risk_level"
            ]
            .astype(str)
            .str.upper()
            == "HIGH"
        )

        .sum()

    )


    multi_engine_count = int(

        (
            investigation_df[
                "active_risk_engines"
            ]
            >= 2
        )

        .sum()

    )


    four_engine_count = int(

        (
            investigation_df[
                "active_risk_engines"
            ]
            >= 4
        )

        .sum()

    )


    summary_rows.append({

        "metric":
            "Total Investigation Candidates",

        "value":
            total_projects

    })


    summary_rows.append({

        "metric":
            "Critical Projects",

        "value":
            critical_count

    })


    summary_rows.append({

        "metric":
            "High Projects",

        "value":
            high_count

    })


    summary_rows.append({

        "metric":
            "Multi-Engine Projects",

        "value":
            multi_engine_count

    })


    summary_rows.append({

        "metric":
            "Four-Engine Projects",

        "value":
            four_engine_count

    })


    summary_rows.append({

        "metric":
            "Average AI Risk Score",

        "value":
            round(

                investigation_df[
                    "final_ai_risk_score"
                ]
                .mean(),

                2

            )

    })


    summary_rows.append({

        "metric":
            "Maximum AI Risk Score",

        "value":
            round(

                investigation_df[
                    "final_ai_risk_score"
                ]
                .max(),

                2

            )

    })


    investigation_summary_df = pd.DataFrame(
        summary_rows
    )


    investigation_summary_df.to_csv(

        SUMMARY_OUTPUT_PATH,

        index=False

    )


    print(
        "\nInvestigation summary saved:"
    )

    print(
        SUMMARY_OUTPUT_PATH
    )


    # ========================================================
    # DISPLAY TOP INVESTIGATION CANDIDATES
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP INVESTIGATION CANDIDATES"
    )

    print(
        "=" * 70
    )


    display_columns = [

        "investigation_rank",

        "work_id",

        "state",

        "constituency",

        "final_ai_risk_score",

        "final_ai_risk_level",

        "active_risk_engines",

        "primary_risk_source",

        "investigation_priority_score",

        "investigation_priority_category"

    ]


    display_columns = [

        column

        for column in display_columns

        if column in investigation_df.columns

    ]


    print(

        investigation_df[
            display_columns
        ]
        .head(30)
        .to_string(
            index=False
        )

    )


    # ========================================================
    # SELECT PROJECTS FOR LLM REPORTING
    # ========================================================

    llm_projects = (

        investigation_df

        .head(
            MAX_LLM_REPORTS
        )

        .copy()

    )


    print(
        "\n" + "=" * 70
    )

    print(
        "GROUNDED LLM INVESTIGATION REPORTING"
    )

    print(
        "=" * 70
    )


    print(
        f"\nProjects selected for LLM reports: "
        f"{len(llm_projects)}"
    )


    if llm is None:

        print(
            "\nLLM unavailable."
        )

        print(
            "Generating deterministic investigation "
            "records only."
        )


    # ========================================================
    # GENERATE REPORTS
    # ========================================================

    os.makedirs(

        REPORT_DIRECTORY,

        exist_ok=True

    )


    report_rows = []


    for position, (_, row) in enumerate(

        llm_projects.iterrows(),

        start=1

    ):

        work_id = row[
            "work_id"
        ]


        print(
            f"\n[{position}/{len(llm_projects)}] "
            f"Generating report: {work_id}"
        )


        try:

            report = generate_grounded_report(
                row
            )


            status = "SUCCESS"


        except Exception as e:

            print(
                f"LLM ERROR: {e}"
            )


            report = (

                "LLM report generation failed.\n\n"

                "Error: "

                + str(e)

            )


            status = "FAILED"


        # ----------------------------------------------------
        # Create markdown report
        # ----------------------------------------------------

        markdown_report = create_markdown_report(

            row,

            report

        )


        filename = (

            clean_filename(
                work_id
            )

            +

            ".md"

        )


        report_path = os.path.join(

            REPORT_DIRECTORY,

            filename

        )


        with open(

            report_path,

            "w",

            encoding="utf-8"

        ) as file:

            file.write(
                markdown_report
            )


        # ----------------------------------------------------
        # Store report information
        # ----------------------------------------------------

        report_rows.append({

            "investigation_rank":
                row[
                    "investigation_rank"
                ],

            "work_id":
                work_id,

            "state":
                get_column(
                    row,
                    "state"
                ),

            "constituency":
                get_column(
                    row,
                    "constituency"
                ),

            "final_ai_risk_score":
                get_column(
                    row,
                    "final_ai_risk_score"
                ),

            "final_ai_risk_level":
                get_column(
                    row,
                    "final_ai_risk_level"
                ),

            "risk_detection_confidence":
                get_column(
                    row,
                    "risk_detection_confidence"
                ),

            "active_risk_engines":
                get_column(
                    row,
                    "active_risk_engines"
                ),

            "available_risk_engines":
                get_column(
                    row,
                    "available_risk_engines"
                ),

            "primary_risk_source":
                get_column(
                    row,
                    "primary_risk_source"
                ),

            "detecting_engines":
                get_column(
                    row,
                    "detecting_engines"
                ),

            "investigation_priority_score":
                get_column(
                    row,
                    "investigation_priority_score"
                ),

            "investigation_priority_category":
                get_column(
                    row,
                    "investigation_priority_category"
                ),

            "report_status":
                status,

            "report_file":
                report_path,

            "grounded_llm_investigation_report":
                report

        })


        # ----------------------------------------------------
        # Small delay to avoid overly aggressive requests
        # ----------------------------------------------------

        if (
            llm is not None
            and
            position < len(llm_projects)
        ):

            time.sleep(
                0.5
            )


    # ========================================================
    # SAVE LLM REPORT DATA
    # ========================================================

    reports_df = pd.DataFrame(
        report_rows
    )


    reports_df.to_csv(

        LLM_OUTPUT_PATH,

        index=False

    )


    print(
        "\nLLM report dataset saved:"
    )

    print(
        LLM_OUTPUT_PATH
    )


    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    successful_reports = int(

        (
            reports_df[
                "report_status"
            ]
            == "SUCCESS"
        )

        .sum()

    ) if len(
        reports_df
    ) > 0 else 0


    failed_reports = int(

        (
            reports_df[
                "report_status"
            ]
            == "FAILED"
        )

        .sum()

    ) if len(
        reports_df
    ) > 0 else 0


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "INVESTIGATION & REPORTING LAYER COMPLETE"
    )

    print(
        "=" * 70
    )


    print(
        f"\nTotal projects in unified dataset: "
        f"{len(df)}"
    )


    print(
        f"HIGH + CRITICAL candidates: "
        f"{len(investigation_df)}"
    )


    print(
        f"Critical candidates: "
        f"{critical_count}"
    )


    print(
        f"High candidates: "
        f"{high_count}"
    )


    print(
        f"Multi-engine candidates: "
        f"{multi_engine_count}"
    )


    print(
        f"Four-engine candidates: "
        f"{four_engine_count}"
    )


    print(
        f"LLM reports attempted: "
        f"{len(llm_projects)}"
    )


    print(
        f"LLM reports successful: "
        f"{successful_reports}"
    )


    print(
        f"LLM reports failed: "
        f"{failed_reports}"
    )


    print(
        "\nOUTPUT FILES"
    )


    print(
        "1.",
        QUEUE_OUTPUT_PATH
    )


    print(
        "2.",
        SUMMARY_OUTPUT_PATH
    )


    print(
        "3.",
        LLM_OUTPUT_PATH
    )


    print(
        "4.",
        REPORT_DIRECTORY
    )


    print(
        "\n" + "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()