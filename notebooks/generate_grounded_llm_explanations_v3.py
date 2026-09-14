import os
import json
import time
import ast
import math
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please add it to your .env file."
    )


# ============================================================
# FILE PATHS
# ============================================================

MASTER_DATASET_PATH = (
    "data/processed/master_dataset.csv"
)

UNIFIED_RISK_PATH = (
    "data/processed/final_unified_risk_analysis_v2.csv"
)

FINANCIAL_RISK_PATH = (
    "data/processed/financial_risk_analysis.csv"
)

STATISTICAL_PATH = (
    "data/processed/statistical_financial_anomaly_analysis.csv"
)

INVESTIGATION_REPORT_PATH = (
    "data/processed/project_investigation_reports.csv"
)

OUTPUT_PATH = (
    "data/processed/grounded_llm_investigation_reports_v3.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_PROJECTS = 20

MODEL_NAME = "openai/gpt-oss-20b"

TEMPERATURE = 0.2

REQUEST_DELAY = 1


# ============================================================
# LLM INITIALIZATION
# ============================================================

print("\nInitializing Groq LLM...")

llm = ChatGroq(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=TEMPERATURE
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_missing(value):
    """
    Safely check whether a value is missing.
    Handles NaN, None, empty strings and numpy values.
    """

    if value is None:
        return True

    if isinstance(value, str):
        return value.strip() == ""

    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    return False


def clean_value(value):
    """
    Convert numpy/pandas values into JSON-safe Python values.
    """

    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.Timestamp):
        return str(value)

    if isinstance(value, float):

        if math.isnan(value):
            return None

        if math.isinf(value):
            return None

        return round(value, 4)

    return value


def get_value(row, column, default="Not available"):
    """
    Safely get a column value from a pandas row.
    """

    if column not in row.index:
        return default

    value = row[column]

    if is_missing(value):
        return default

    return clean_value(value)


def get_first_available(row, columns, default="Not available"):
    """
    Return the first valid value from multiple possible columns.
    """

    for column in columns:

        if column in row.index:

            value = row[column]

            if not is_missing(value):
                return clean_value(value)

    return default


def parse_list_value(value):
    """
    Convert list-like strings safely into Python lists.

    Handles:
    - Python lists
    - numpy arrays
    - NaN
    - string representations of lists
    - normal strings
    """

    if value is None:
        return []

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    if isinstance(value, str):

        value = value.strip()

        if not value:
            return []

        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                return parsed

            return [str(parsed)]

        except Exception:
            return [value]

    return [str(value)]


def format_currency(value):
    """
    Format financial values in Indian Rupees.
    """

    if value is None:
        return "Not available"

    try:

        if isinstance(value, str):

            if value == "Not available":
                return value

            value = float(value)

        if pd.isna(value):
            return "Not available"

        return f"₹{value:,.2f}"

    except Exception:

        return str(value)


def format_number(value):
    """
    Format numeric values safely.
    """

    if value is None:
        return "Not available"

    try:

        if isinstance(value, str):

            if value == "Not available":
                return value

            value = float(value)

        if pd.isna(value):
            return "Not available"

        return round(float(value), 4)

    except Exception:

        return str(value)


def safe_json_dumps(data):
    """
    Convert evidence dictionary into safe JSON.
    """

    cleaned = {}

    for key, value in data.items():

        if isinstance(value, dict):

            cleaned[key] = {
                k: clean_value(v)
                for k, v in value.items()
            }

        elif isinstance(value, list):

            cleaned[key] = [
                clean_value(item)
                for item in value
            ]

        else:

            cleaned[key] = clean_value(value)

    return json.dumps(
        cleaned,
        indent=2,
        ensure_ascii=False,
        default=str
    )


# ============================================================
# LOAD DATASETS
# ============================================================

print("\n============================================================")
print("LOADING GROUNDED INVESTIGATION DATASETS")
print("============================================================")


print("\nLoading master dataset...")

master_df = pd.read_csv(
    MASTER_DATASET_PATH
)

print(f"Master projects: {len(master_df)}")


print("\nLoading unified risk analysis...")

unified_df = pd.read_csv(
    UNIFIED_RISK_PATH
)

print(f"Unified projects: {len(unified_df)}")


print("\nLoading financial risk analysis...")

financial_df = pd.read_csv(
    FINANCIAL_RISK_PATH
)

print(f"Financial projects: {len(financial_df)}")


print("\nLoading statistical anomaly analysis...")

statistical_df = pd.read_csv(
    STATISTICAL_PATH
)

print(f"Statistical projects: {len(statistical_df)}")


print("\nLoading investigation reports...")

investigation_df = pd.read_csv(
    INVESTIGATION_REPORT_PATH
)

print(f"Investigation reports: {len(investigation_df)}")


# ============================================================
# VALIDATE WORK_ID
# ============================================================

datasets = {
    "master_dataset": master_df,
    "unified_risk": unified_df,
    "financial_risk": financial_df,
    "statistical_analysis": statistical_df,
    "investigation_reports": investigation_df
}


print("\n============================================================")
print("VALIDATING DATASETS")
print("============================================================")


for name, df in datasets.items():

    if "work_id" not in df.columns:

        raise ValueError(
            f"'work_id' column missing in {name}"
        )

    df["work_id"] = df["work_id"].astype(str).str.strip()


# ============================================================
# REMOVE DUPLICATE WORK IDS
# ============================================================

print("\nRemoving duplicate work_id records if any...")


for name, df in datasets.items():

    before = len(df)

    df.drop_duplicates(
        subset=["work_id"],
        keep="first",
        inplace=True
    )

    after = len(df)

    if before != after:

        print(
            f"{name}: removed {before - after} duplicate records"
        )


# ============================================================
# SELECT IMPORTANT COLUMNS
# ============================================================

print("\nPreparing datasets for merge...")


# ------------------------------------------------------------
# MASTER DATASET
# ------------------------------------------------------------

master_columns = [
    "work_id",
    "state",
    "ida",
    "honble_members_of_parliament",
    "constituency",
    "work_category",
    "work_description",
    "recommended_date",
    "recommended_amount",
    "sanction_date",
    "sanction_amount",
    "work_status",
    "completion_date",
    "completed_amount_disbursed",
    "total_expenditure",
    "payment_count",
    "unique_vendors",
    "payments_per_vendor",
    "has_sanction",
    "has_completion",
    "has_expenditure",
    "sanction_delay_days",
    "completion_duration_days",
    "sanction_difference",
    "expenditure_ratio",
    "sanction_ratio",
    "expenditure_difference",
    "completion_amount_ratio",
    "completion_amount_difference",
    "financial_year",
    "recommended_year",
    "recommended_month"
]


master_columns = [
    column
    for column in master_columns
    if column in master_df.columns
]


master_clean = master_df[
    master_columns
].copy()


# ------------------------------------------------------------
# UNIFIED RISK DATASET
# ------------------------------------------------------------

unified_columns = [
    "work_id",
    "combined_anomaly_score",
    "ensemble_risk_level",
    "rule_risk_score",
    "rule_risk_level",
    "risk_factors",
    "recommended_actions",
    "financial_risk_score",
    "financial_risk_level",
    "financial_risk_factors",
    "statistical_anomaly_score",
    "statistical_anomaly_level",
    "statistical_anomaly_factors",
    "ml_normalized_score",
    "final_ai_risk_score",
    "final_ai_risk_level",
    "active_risk_engines",
    "final_ai_risk_factors",
    "risk_detection_confidence",
    "primary_risk_source",
    "consensus_bonus"
]


unified_columns = [
    column
    for column in unified_columns
    if column in unified_df.columns
]


unified_clean = unified_df[
    unified_columns
].copy()


# ------------------------------------------------------------
# FINANCIAL DATASET
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


financial_clean = financial_df[
    financial_columns
].copy()


# Rename to avoid conflicts

financial_rename_map = {}

for column in financial_clean.columns:

    if column != "work_id":

        financial_rename_map[column] = (
            f"{column}_financial_engine"
        )


financial_clean.rename(
    columns=financial_rename_map,
    inplace=True
)


# ------------------------------------------------------------
# STATISTICAL DATASET
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


statistical_clean = statistical_df[
    statistical_columns
].copy()


# Rename to avoid conflicts

statistical_rename_map = {}

for column in statistical_clean.columns:

    if column != "work_id":

        statistical_rename_map[column] = (
            f"{column}_statistical_engine"
        )


statistical_clean.rename(
    columns=statistical_rename_map,
    inplace=True
)


# ------------------------------------------------------------
# INVESTIGATION DATASET
# ------------------------------------------------------------

investigation_columns = [
    "work_id",
    "investigation_priority",
    "investigation_summary",
    "key_risk_indicators",
    "recommended_actions",
    "risk_factors"
]


investigation_columns = [
    column
    for column in investigation_columns
    if column in investigation_df.columns
]


investigation_clean = investigation_df[
    investigation_columns
].copy()


# Rename conflicting columns

investigation_rename_map = {}

for column in investigation_clean.columns:

    if column != "work_id":

        investigation_rename_map[column] = (
            f"{column}_investigation"
        )


investigation_clean.rename(
    columns=investigation_rename_map,
    inplace=True
)


# ============================================================
# MERGE DATASETS
# ============================================================

print("\n============================================================")
print("MERGING VERIFIED RISK EVIDENCE")
print("============================================================")


# Start with unified risk dataset

merged_df = unified_clean.copy()


# Merge original project information

merged_df = merged_df.merge(
    master_clean,
    on="work_id",
    how="left"
)


# Merge financial engine

merged_df = merged_df.merge(
    financial_clean,
    on="work_id",
    how="left"
)


# Merge statistical engine

merged_df = merged_df.merge(
    statistical_clean,
    on="work_id",
    how="left"
)


# Merge investigation reports

merged_df = merged_df.merge(
    investigation_clean,
    on="work_id",
    how="left"
)


print(f"\nMerged projects: {len(merged_df)}")


# ============================================================
# DATA AVAILABILITY CHECK
# ============================================================

print("\n============================================================")
print("IMPORTANT COLUMN AVAILABILITY")
print("============================================================")


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


for column in important_columns:

    print(
        f"{column}: {column in merged_df.columns}"
    )


# ============================================================
# BUILD GROUNDED EVIDENCE
# ============================================================

def build_project_evidence(row):

    # --------------------------------------------------------
    # PROJECT INFORMATION
    # --------------------------------------------------------

    project_info = {

        "work_id": get_value(row, "work_id"),

        "state": get_value(row, "state"),

        "ida": get_value(row, "ida"),

        "member_of_parliament": get_value(
            row,
            "honble_members_of_parliament"
        ),

        "constituency": get_value(
            row,
            "constituency"
        ),

        "work_category": get_value(
            row,
            "work_category"
        ),

        "work_description": get_value(
            row,
            "work_description"
        ),

        "work_status": get_value(
            row,
            "work_status"
        ),

        "financial_year": get_value(
            row,
            "financial_year"
        )
    }


    # --------------------------------------------------------
    # FINANCIAL INFORMATION
    # --------------------------------------------------------

    recommended_amount = get_value(
        row,
        "recommended_amount"
    )

    sanction_amount = get_value(
        row,
        "sanction_amount"
    )

    total_expenditure = get_value(
        row,
        "total_expenditure"
    )


    financial_info = {

        "recommended_amount": format_currency(
            recommended_amount
        ),

        "sanction_amount": format_currency(
            sanction_amount
        ),

        "total_expenditure": format_currency(
            total_expenditure
        ),

        "completed_amount_disbursed": format_currency(
            get_value(
                row,
                "completed_amount_disbursed"
            )
        ),

        "sanction_difference": format_currency(
            get_value(
                row,
                "sanction_difference"
            )
        ),

        "expenditure_difference": format_currency(
            get_value(
                row,
                "expenditure_difference"
            )
        ),

        "expenditure_ratio": format_number(
            get_value(
                row,
                "expenditure_ratio"
            )
        ),

        "sanction_ratio": format_number(
            get_value(
                row,
                "sanction_ratio"
            )
        )
    }


    # --------------------------------------------------------
    # PAYMENT INFORMATION
    # --------------------------------------------------------

    payment_info = {

        "payment_count": get_value(
            row,
            "payment_count"
        ),

        "unique_vendors": get_value(
            row,
            "unique_vendors"
        ),

        "payments_per_vendor": get_value(
            row,
            "payments_per_vendor"
        )
    }


    # --------------------------------------------------------
    # UNIFIED AI RISK ENGINE
    # --------------------------------------------------------

    unified_risk = {

        "final_ai_risk_score": get_value(
            row,
            "final_ai_risk_score"
        ),

        "final_ai_risk_level": get_value(
            row,
            "final_ai_risk_level"
        ),

        "risk_detection_confidence": get_value(
            row,
            "risk_detection_confidence"
        ),

        "active_risk_engines": get_value(
            row,
            "active_risk_engines"
        ),

        "primary_risk_source": get_value(
            row,
            "primary_risk_source"
        ),

        "consensus_bonus": get_value(
            row,
            "consensus_bonus"
        ),

        "final_ai_risk_factors": parse_list_value(
            get_value(
                row,
                "final_ai_risk_factors",
                None
            )
        )
    }


    # --------------------------------------------------------
    # MACHINE LEARNING ENGINE
    # --------------------------------------------------------

    ml_engine = {

        "combined_anomaly_score": get_value(
            row,
            "combined_anomaly_score"
        ),

        "ensemble_risk_level": get_value(
            row,
            "ensemble_risk_level"
        ),

        "ml_normalized_score": get_value(
            row,
            "ml_normalized_score"
        )
    }


    # --------------------------------------------------------
    # RULE-BASED ENGINE
    # --------------------------------------------------------

    rule_engine = {

        "rule_risk_score": get_value(
            row,
            "rule_risk_score"
        ),

        "rule_risk_level": get_value(
            row,
            "rule_risk_level"
        ),

        "risk_factors": parse_list_value(
            get_value(
                row,
                "risk_factors",
                None
            )
        ),

        "recommended_actions": parse_list_value(
            get_value(
                row,
                "recommended_actions",
                None
            )
        )
    }


    # --------------------------------------------------------
    # FINANCIAL RISK ENGINE
    # --------------------------------------------------------

    financial_engine = {

        "financial_risk_score": get_first_available(
            row,
            [
                "financial_risk_score_financial_engine",
                "financial_risk_score"
            ]
        ),

        "financial_risk_level": get_first_available(
            row,
            [
                "financial_risk_level_financial_engine",
                "financial_risk_level"
            ]
        ),

        "financial_risk_factors": parse_list_value(
            get_first_available(
                row,
                [
                    "financial_risk_factors_financial_engine",
                    "financial_risk_factors"
                ],
                default=None
            )
        )
    }


    # --------------------------------------------------------
    # STATISTICAL ANOMALY ENGINE
    # --------------------------------------------------------

    statistical_engine = {

        "statistical_anomaly_score": get_first_available(
            row,
            [
                "statistical_anomaly_score_statistical_engine",
                "statistical_anomaly_score"
            ]
        ),

        "statistical_anomaly_level": get_first_available(
            row,
            [
                "statistical_anomaly_level_statistical_engine",
                "statistical_anomaly_level"
            ]
        ),

        "statistical_anomaly_factors": parse_list_value(
            get_first_available(
                row,
                [
                    "statistical_anomaly_factors_statistical_engine",
                    "statistical_anomaly_factors"
                ],
                default=None
            )
        )
    }


    # --------------------------------------------------------
    # INVESTIGATION ENGINE
    # --------------------------------------------------------

    investigation = {

        "investigation_priority": get_value(
            row,
            "investigation_priority_investigation"
        ),

        "investigation_summary": get_value(
            row,
            "investigation_summary_investigation"
        ),

        "key_risk_indicators": parse_list_value(
            get_value(
                row,
                "key_risk_indicators_investigation",
                None
            )
        ),

        "recommended_actions": parse_list_value(
            get_value(
                row,
                "recommended_actions_investigation",
                None
            )
        ),

        "risk_factors": parse_list_value(
            get_value(
                row,
                "risk_factors_investigation",
                None
            )
        )
    }


    # --------------------------------------------------------
    # FINAL EVIDENCE PACKAGE
    # --------------------------------------------------------

    evidence = {

        "PROJECT_INFORMATION": project_info,

        "FINANCIAL_INFORMATION": financial_info,

        "PAYMENT_INFORMATION": payment_info,

        "UNIFIED_AI_RISK_ENGINE": unified_risk,

        "MACHINE_LEARNING_ENGINE": ml_engine,

        "RULE_BASED_RISK_ENGINE": rule_engine,

        "FINANCIAL_RISK_ENGINE": financial_engine,

        "STATISTICAL_ANOMALY_ENGINE": statistical_engine,

        "INVESTIGATION_ENGINE": investigation
    }


    return evidence


# ============================================================
# PROJECT SELECTION
# ============================================================

print("\n============================================================")
print("SELECTING PROJECTS FOR GROUNDED INVESTIGATION")
print("============================================================")


# Ensure score exists

if "final_ai_risk_score" in merged_df.columns:

    merged_df["final_ai_risk_score"] = pd.to_numeric(
        merged_df["final_ai_risk_score"],
        errors="coerce"
    ).fillna(0)

else:

    merged_df["final_ai_risk_score"] = 0


# Sort by highest risk

selected_projects = merged_df.sort_values(
    by="final_ai_risk_score",
    ascending=False
).head(TOP_PROJECTS).copy()


print(
    f"\nProjects selected: {len(selected_projects)}"
)


# ============================================================
# BUILD GROUNDED PROMPT
# ============================================================

def build_prompt(evidence):

    evidence_json = safe_json_dumps(evidence)

    prompt = f"""
You are an AI Investigation Copilot for a Government Project Risk Intelligence System.

Your task is to generate a professional investigation report using ONLY the verified evidence provided below.

IMPORTANT GROUNDING RULES:

1. DO NOT invent facts.
2. DO NOT assume missing information.
3. Use ONLY the evidence provided.
4. If information is unavailable, explicitly say:
   "Not available in the provided evidence."
5. Do NOT claim corruption, fraud, or wrongdoing.
6. Risk scores and anomalies are indicators requiring verification.
7. Clearly distinguish:
   - verified observations
   - AI risk indicators
   - recommended investigation actions
8. Do not exaggerate conclusions.
9. Financial values should be interpreted carefully.
10. The report must be useful for government auditors and investigators.

VERIFIED PROJECT EVIDENCE:

{evidence_json}

Generate the investigation report using EXACTLY the following structure:

# EXECUTIVE SUMMARY

Provide a concise summary explaining the overall AI risk classification and why the project requires attention.

# PROJECT OVERVIEW

Include:
- Work ID
- State
- Constituency
- Work category
- Work description
- Work status
- Financial year

# RISK CLASSIFICATION

Include:
- Final AI risk level
- Final AI risk score
- Detection confidence
- Number of active risk engines
- Primary risk source
- Consensus information if available

# KEY VERIFIED RISK INDICATORS

List the strongest verified signals from:
- Unified AI risk engine
- ML engine
- Rule-based engine
- Financial risk engine
- Statistical anomaly engine

Only include signals actually available in the evidence.

# FINANCIAL OBSERVATIONS

Analyze:
- Recommended amount
- Sanction amount
- Total expenditure
- Expenditure differences
- Relevant ratios

Do not infer wrongdoing.

# PAYMENT OBSERVATIONS

Analyze:
- Payment count
- Unique vendors
- Payments per vendor

Explain whether the evidence indicates an unusual pattern, but only if supported by the risk-engine evidence.

# STATISTICAL ANOMALY OBSERVATIONS

Explain:
- Statistical anomaly score
- Statistical anomaly level
- Verified statistical anomaly factors

# MACHINE LEARNING OBSERVATIONS

Explain:
- Ensemble risk level
- Combined anomaly score
- ML normalized score

# WHY THIS PROJECT WAS FLAGGED

Explain clearly how the available risk engines contributed to the final risk classification.

Do not invent connections between engines.

# INVESTIGATION PRIORITIES

Provide:

### HIGH PRIORITY
Only actions strongly supported by the evidence.

### MEDIUM PRIORITY
Verification activities that should follow.

### LOW PRIORITY
Additional monitoring or contextual analysis.

# RECOMMENDED NEXT STEPS

Provide practical actions for investigators.

Examples may include:
- Verify financial records
- Review sanction documentation
- Audit payment records
- Compare with similar projects
- Validate supporting documentation

Only recommend actions relevant to the evidence.

# FINAL ASSESSMENT

Summarize the risk profile.

Clearly state:

"The detected anomalies are risk indicators requiring verification and do not independently establish wrongdoing."

Make the report professional, factual, structured, and concise.
"""

    return prompt


# ============================================================
# GENERATE LLM REPORT
# ============================================================

def generate_llm_report(evidence):

    try:

        prompt = build_prompt(evidence)

        response = llm.invoke(
            [
                HumanMessage(
                    content=prompt
                )
            ]
        )

        return response.content

    except Exception as error:

        print(f"\nLLM Error: {error}")

        return (
            "LLM explanation could not be generated. "
            f"Error: {str(error)}"
        )


# ============================================================
# GENERATE REPORTS
# ============================================================

print("\n============================================================")
print("GENERATING GROUNDED LLM INVESTIGATION REPORTS V3")
print("============================================================")


results = []


for index, row in selected_projects.iterrows():

    work_id = get_value(
        row,
        "work_id"
    )

    print(
        f"\nGenerating report for: {work_id}"
    )


    # Build verified evidence

    evidence = build_project_evidence(row)


    # Generate LLM report

    llm_report = generate_llm_report(
        evidence
    )


    # Save result

    results.append({

        "work_id": work_id,

        "final_ai_risk_score": get_value(
            row,
            "final_ai_risk_score"
        ),

        "final_ai_risk_level": get_value(
            row,
            "final_ai_risk_level"
        ),

        "risk_detection_confidence": get_value(
            row,
            "risk_detection_confidence"
        ),

        "active_risk_engines": get_value(
            row,
            "active_risk_engines"
        ),

        "state": get_value(
            row,
            "state"
        ),

        "constituency": get_value(
            row,
            "constituency"
        ),

        "work_category": get_value(
            row,
            "work_category"
        ),

        "work_status": get_value(
            row,
            "work_status"
        ),

        "recommended_amount": get_value(
            row,
            "recommended_amount"
        ),

        "sanction_amount": get_value(
            row,
            "sanction_amount"
        ),

        "total_expenditure": get_value(
            row,
            "total_expenditure"
        ),

        "payment_count": get_value(
            row,
            "payment_count"
        ),

        "unique_vendors": get_value(
            row,
            "unique_vendors"
        ),

        "payments_per_vendor": get_value(
            row,
            "payments_per_vendor"
        ),

        "grounded_evidence": safe_json_dumps(
            evidence
        ),

        "grounded_llm_report": llm_report
    })


    print(
        "Report generated successfully."
    )


    time.sleep(
        REQUEST_DELAY
    )


# ============================================================
# CREATE OUTPUT DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n============================================================")
print("SAVING RESULTS")
print("============================================================")


os.makedirs(
    "data/processed",
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n============================================================")
print("GROUNDED LLM INVESTIGATION COPILOT V3 COMPLETE")
print("============================================================")


print(
    f"\nReports generated: {len(results_df)}"
)


print(
    f"\nResults saved successfully!\n{OUTPUT_PATH}"
)


# ============================================================
# SAMPLE REPORT
# ============================================================

if len(results_df) > 0:

    print("\n============================================================")
    print("SAMPLE GROUNDED INVESTIGATION REPORT")
    print("============================================================\n")

    print(
        results_df.iloc[0][
            "grounded_llm_report"
        ]
    )


print("\n============================================================")
print("PROGRAM COMPLETED SUCCESSFULLY")
print("============================================================")