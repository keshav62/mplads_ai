import os
import json
import ast
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from langchain_groq import ChatGroq
# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Please add it to your .env file."
    )


# ============================================================
# INITIALIZE GROQ LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_missing(value):
    """
    Safely check whether a value is missing.
    Works with strings, numbers, lists, NumPy arrays, and Pandas values.
    """

    if value is None:
        return True

    # NumPy arrays should not be checked using normal boolean logic
    if isinstance(value, np.ndarray):
        return False

    # Lists and dictionaries are valid values
    if isinstance(value, (list, dict)):
        return False

    try:
        result = pd.isna(value)

        if isinstance(result, (bool, np.bool_)):
            return bool(result)

        return False

    except Exception:
        return False

def safe_value(value, default="Not available"):
    """
    Convert missing values into a readable format.
    """

    if is_missing(value):
        return default

    # Convert NumPy values to normal Python values
    if isinstance(value, np.generic):
        return value.item()

    # Convert NumPy arrays to Python lists
    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def parse_list(value):
    """
    Safely convert values into a Python list.
    """

    if is_missing(value):
        return []

    # NumPy array
    if isinstance(value, np.ndarray):
        return value.tolist()

    # Already a list
    if isinstance(value, list):
        return value

    # Tuple
    if isinstance(value, tuple):
        return list(value)

    # String
    if isinstance(value, str):

        value = value.strip()

        if not value:
            return []

        try:

            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                return parsed

            if isinstance(parsed, tuple):
                return list(parsed)

        except Exception:
            pass

        return [value]

    return [safe_value(value)]

def get_column(row, column_name, default="Not available"):
    """
    Safely retrieve a column from a Pandas row.
    """

    if column_name not in row.index:
        return default

    value = row[column_name]

    if is_missing(value):
        return default

    return safe_value(value, default)


# ============================================================
# BUILD VERIFIED PROJECT EVIDENCE
# ============================================================

def build_project_evidence(project):

    evidence = {

        # ----------------------------------------------------
        # PROJECT INFORMATION
        # ----------------------------------------------------

        "project_information": {

            "work_id": get_column(
                project,
                "work_id"
            ),

            "state": get_column(
                project,
                "state"
            ),

            "constituency": get_column(
                project,
                "constituency"
            ),

            "work_category": get_column(
                project,
                "work_category"
            ),

            "work_description": get_column(
                project,
                "work_description"
            ),

            "work_status": get_column(
                project,
                "work_status"
            )
        },


        # ----------------------------------------------------
        # FINANCIAL INFORMATION
        # ----------------------------------------------------

        "financial_information": {

            "recommended_amount": get_column(
                project,
                "recommended_amount"
            ),

            "sanction_amount": get_column(
                project,
                "sanction_amount"
            ),

            "total_expenditure": get_column(
                project,
                "total_expenditure"
            ),

            "completed_amount_disbursed": get_column(
                project,
                "completed_amount_disbursed"
            ),

            "expenditure_ratio": get_column(
                project,
                "expenditure_ratio"
            ),

            "sanction_ratio": get_column(
                project,
                "sanction_ratio"
            )
        },


        # ----------------------------------------------------
        # PAYMENT INFORMATION
        # ----------------------------------------------------

        "payment_information": {

            "payment_count": get_column(
                project,
                "payment_count"
            ),

            "unique_vendors": get_column(
                project,
                "unique_vendors"
            ),

            "payments_per_vendor": get_column(
                project,
                "payments_per_vendor"
            )
        },


        # ----------------------------------------------------
        # UNIFIED AI RISK ENGINE
        # ----------------------------------------------------

        "unified_ai_risk": {

            "final_ai_risk_score": get_column(
                project,
                "final_ai_risk_score"
            ),

            "final_ai_risk_level": get_column(
                project,
                "final_ai_risk_level"
            ),

            "risk_detection_confidence": get_column(
                project,
                "risk_detection_confidence"
            ),

            "active_risk_engines": get_column(
                project,
                "active_risk_engines"
            ),

            "consensus_bonus": get_column(
                project,
                "consensus_bonus"
            ),

            "primary_risk_source": get_column(
                project,
                "primary_risk_source"
            )
        },


        # ----------------------------------------------------
        # FINANCIAL RISK ENGINE
        # ----------------------------------------------------

        "financial_risk_engine": {

            "financial_risk_score": get_column(
                project,
                "financial_risk_score"
            ),

            "financial_risk_level": get_column(
                project,
                "financial_risk_level"
            ),

            "financial_risk_factors": parse_list(
                get_column(
                    project,
                    "financial_risk_factors",
                    []
                )
            )
        },


        # ----------------------------------------------------
        # STATISTICAL ANOMALY ENGINE
        # ----------------------------------------------------

        "statistical_anomaly_engine": {

            "statistical_anomaly_score": get_column(
                project,
                "statistical_anomaly_score"
            ),

            "statistical_anomaly_level": get_column(
                project,
                "statistical_anomaly_level"
            ),

            "statistical_anomaly_factors": parse_list(
                get_column(
                    project,
                    "statistical_anomaly_factors",
                    []
                )
            )
        },


        # ----------------------------------------------------
        # ML / ANOMALY ENGINE
        # ----------------------------------------------------

        "machine_learning_engine": {

            "ensemble_risk_level": get_column(
                project,
                "ensemble_risk_level"
            ),

            "combined_anomaly_score": get_column(
                project,
                "combined_anomaly_score"
            ),

            "ensemble_is_anomaly": get_column(
                project,
                "ensemble_is_anomaly"
            )
        },


        # ----------------------------------------------------
        # PREVIOUS INVESTIGATION INFORMATION
        # ----------------------------------------------------

        "existing_investigation_data": {

            "risk_factors": parse_list(
                get_column(
                    project,
                    "risk_factors",
                    []
                )
            ),

            "recommended_actions": parse_list(
                get_column(
                    project,
                    "recommended_actions",
                    []
                )
            )
        }
    }

    return evidence


# ============================================================
# GENERATE GROUNDED LLM REPORT
# ============================================================

def generate_grounded_report(project):

    evidence = build_project_evidence(project)

    prompt = f"""
You are an AI Investigation Copilot for a Government Project
Risk Intelligence System.

Your responsibility is to generate a professional investigation
report using ONLY the VERIFIED PROJECT EVIDENCE provided below.

============================================================
IMPORTANT GROUNDING RULES
============================================================

1. Use ONLY the information provided in the evidence.

2. NEVER invent:
   - financial amounts
   - percentages
   - vendors
   - transactions
   - corruption allegations
   - fraud allegations
   - missing risk signals

3. NEVER claim that fraud, corruption, misuse of funds, or illegal
activity has occurred.

4. Risk scores and anomalies indicate potential areas requiring
review. They are NOT proof of wrongdoing.

5. If information is missing, write:

"Not available in the provided evidence."

6. Do not modify numerical values.

7. Clearly distinguish between:

   - Verified Observation
   - Risk Indicator
   - AI/ML Detection
   - Investigation Recommendation

8. Do not exaggerate the severity of the project.

9. If a project is flagged by multiple engines, explain that this
represents multiple independent risk signals, not confirmation of
wrongdoing.

10. Write for government auditors and investigators.

============================================================
VERIFIED PROJECT EVIDENCE
============================================================

{json.dumps(evidence, indent=2, default=str)}

============================================================
REPORT FORMAT
============================================================

Generate the report using EXACTLY these sections:

# EXECUTIVE SUMMARY

Provide a concise explanation of why the project requires attention.

# PROJECT OVERVIEW

Summarize the available project information.

# RISK CLASSIFICATION

Explain:

- Final AI risk level
- Final AI risk score
- Detection confidence
- Active risk engines
- Primary risk source

# KEY VERIFIED RISK INDICATORS

List only the actual risk indicators available in the evidence.

# FINANCIAL OBSERVATIONS

Explain:

- Recommended amount
- Sanction amount
- Total expenditure
- Expenditure relationships
- Financial risk engine findings

Do not calculate new values unless directly possible from the
provided evidence.

# PAYMENT OBSERVATIONS

Explain:

- Payment count
- Unique vendors
- Payments per vendor
- Payment concentration risk indicators

# STATISTICAL ANOMALY OBSERVATIONS

Explain the findings from the Statistical Anomaly Engine.

# MACHINE LEARNING OBSERVATIONS

Explain the findings from the ML / Ensemble Anomaly Engine.

# WHY THIS PROJECT WAS FLAGGED

Explain how the available engines and verified evidence contributed
to the risk assessment.

# INVESTIGATION PRIORITIES

Provide investigation priorities based ONLY on the detected risk
signals.

Use categories such as:

- HIGH PRIORITY
- MEDIUM PRIORITY
- LOW PRIORITY

Only assign these priorities based on available evidence.

# RECOMMENDED NEXT STEPS

Provide practical verification actions.

Examples may include:

- Verify expenditure records
- Review payment documentation
- Compare with similar projects
- Validate sanction compliance

Only recommend actions supported by the risk evidence.

# FINAL ASSESSMENT

Provide a balanced conclusion.

Clearly state that:

"The detected anomalies are risk indicators requiring verification
and do not independently establish wrongdoing."

Keep the report factual, professional, concise, and evidence-based.
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading Grounded Investigation datasets...")


# ------------------------------------------------------------
# UNIFIED RISK DATASET
# ------------------------------------------------------------

unified_df = pd.read_csv(
    "data/processed/final_unified_risk_analysis.csv"
)

print(
    f"Unified projects: {len(unified_df)}"
)


# ------------------------------------------------------------
# FINANCIAL RISK DATASET
# ------------------------------------------------------------

financial_df = pd.read_csv(
    "data/processed/financial_risk_analysis.csv"
)

print(
    f"Financial projects: {len(financial_df)}"
)


# ------------------------------------------------------------
# STATISTICAL ANOMALY DATASET
# ------------------------------------------------------------

statistical_df = pd.read_csv(
    "data/processed/statistical_financial_anomaly_analysis.csv"
)

print(
    f"Statistical projects: {len(statistical_df)}"
)


# ------------------------------------------------------------
# INVESTIGATION REPORT DATASET
# ------------------------------------------------------------

investigation_df = pd.read_csv(
    "data/processed/project_investigation_reports.csv"
)

print(
    f"Investigation reports: {len(investigation_df)}"
)


# ============================================================
# CLEAN DATASETS BEFORE MERGING
# ============================================================

print("\nMerging verified risk evidence...")


# ------------------------------------------------------------
# Select useful columns from financial dataset
# ------------------------------------------------------------

financial_columns = [
    "work_id",
    "financial_risk_score",
    "financial_risk_level",
    "financial_risk_factors"
]

financial_columns = [
    column
    for column in financial_columns
    if column in financial_df.columns
]

financial_df = financial_df[
    financial_columns
]


# ------------------------------------------------------------
# Select useful columns from statistical dataset
# ------------------------------------------------------------

statistical_columns = [
    "work_id",
    "statistical_anomaly_score",
    "statistical_anomaly_level",
    "statistical_anomaly_factors"
]

statistical_columns = [
    column
    for column in statistical_columns
    if column in statistical_df.columns
]

statistical_df = statistical_df[
    statistical_columns
]


# ------------------------------------------------------------
# Select useful investigation columns
# ------------------------------------------------------------

investigation_columns = [
    "work_id",
    "risk_factors",
    "recommended_actions"
]

investigation_columns = [
    column
    for column in investigation_columns
    if column in investigation_df.columns
]

investigation_df = investigation_df[
    investigation_columns
]


# ============================================================
# MERGE DATASETS
# ============================================================

merged_df = unified_df.merge(

    financial_df,

    on="work_id",

    how="left"
)


merged_df = merged_df.merge(

    statistical_df,

    on="work_id",

    how="left"
)


merged_df = merged_df.merge(

    investigation_df,

    on="work_id",

    how="left"
)


print(
    f"Merged projects: {len(merged_df)}"
)

print("\nMERGED DATASET COLUMNS:\n")

for col in merged_df.columns:
    print(col)


print("\nSAMPLE MERGED PROJECT:\n")

print(
    merged_df.iloc[0][
        [
            "work_id"
        ]
    ]
)


important_columns = [
    "work_id",
    "state",
    "constituency",
    "work_category",
    "recommended_amount",
    "sanction_amount",
    "total_expenditure",
    "payment_count",
    "unique_vendors",
    "payments_per_vendor"
]

print("\nIMPORTANT COLUMN AVAILABILITY:\n")

for col in important_columns:
    print(f"{col}: {col in merged_df.columns}")

# ============================================================
# SELECT PROJECTS FOR INVESTIGATION
# ============================================================

print("\nSelecting projects for grounded investigation...")


# ------------------------------------------------------------
# Sort projects by final AI risk score
# ------------------------------------------------------------

if "final_ai_risk_score" in merged_df.columns:

    selected_projects = merged_df.sort_values(

        by="final_ai_risk_score",

        ascending=False

    ).head(20)

else:

    # Fallback if final score does not exist

    selected_projects = merged_df.head(20)


print(
    f"Projects selected: {len(selected_projects)}"
)


# ============================================================
# GENERATE GROUNDED LLM REPORTS
# ============================================================

print(
    "\nGenerating grounded LLM investigation reports..."
)


grounded_reports = []


for index, row in selected_projects.iterrows():

    work_id = row["work_id"]

    print(
        f"\nGenerating report for: {work_id}"
    )

    try:

        report = generate_grounded_report(row)

        print(
            "Report generated successfully."
        )

    except Exception as e:

        print(
            f"LLM Error: {e}"
        )

        report = (
            "LLM explanation could not be generated. "
            f"Error: {str(e)}"
        )

    grounded_reports.append(report)


# ============================================================
# ADD REPORTS TO DATAFRAME
# ============================================================

selected_projects = selected_projects.copy()

selected_projects[
    "grounded_llm_investigation_report"
] = grounded_reports


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    "data/processed/"
    "grounded_llm_investigation_reports_v2.csv"
)


selected_projects.to_csv(

    output_path,

    index=False

)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 60)

print(
    "GROUNDED LLM INVESTIGATION COPILOT V2 COMPLETE"
)

print("=" * 60)


print(
    f"\nReports generated: {len(selected_projects)}"
)


print("\nResults saved successfully!")

print(output_path)


# ============================================================
# SHOW SAMPLE REPORT
# ============================================================

print("\n")
print("=" * 60)

print(
    "SAMPLE GROUNDED INVESTIGATION REPORT"
)

print("=" * 60)


if len(selected_projects) > 0:

    print(
        selected_projects.iloc[0][
            "grounded_llm_investigation_report"
        ]
    )


print("\n")
print("=" * 60)
print("PROGRAM COMPLETED SUCCESSFULLY")
print("=" * 60)