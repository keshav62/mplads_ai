import os
import json
import pandas as pd

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# INITIALIZE LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2
)


# ============================================================
# LOAD INVESTIGATION REPORTS
# ============================================================

print("\nLoading project investigation reports...")

df = pd.read_csv(
    "data/processed/project_investigation_reports.csv"
)

print("Total projects:", len(df))


# ============================================================
# HELPER FUNCTION
# ============================================================

def safe_value(row, column):

    if column not in row.index:
        return "Not Available"

    value = row[column]

    if pd.isna(value):
        return "Not Available"

    return value


# ============================================================
# BUILD PROJECT CONTEXT
# ============================================================

def build_project_context(row):

    context = f"""
PROJECT INFORMATION

Work ID:
{safe_value(row, "work_id")}

State:
{safe_value(row, "state")}

Constituency:
{safe_value(row, "constituency")}

Work Category:
{safe_value(row, "work_category")}

Work Description:
{safe_value(row, "work_description")}


FINANCIAL INFORMATION

Recommended Amount:
{safe_value(row, "recommended_amount")}

Sanction Amount:
{safe_value(row, "sanction_amount")}

Total Expenditure:
{safe_value(row, "total_expenditure")}

Payment Count:
{safe_value(row, "payment_count")}

Unique Vendors:
{safe_value(row, "unique_vendors")}

Payments Per Vendor:
{safe_value(row, "payments_per_vendor")}


AI RISK ANALYSIS

Final AI Risk Score:
{safe_value(row, "final_ai_risk_score")}

Final AI Risk Level:
{safe_value(row, "final_ai_risk_level")}

Risk Detection Confidence:
{safe_value(row, "risk_detection_confidence")}

Active Risk Engines:
{safe_value(row, "active_risk_engines")}

Primary Risk Source:
{safe_value(row, "primary_risk_source")}


FINANCIAL RISK

Financial Risk Score:
{safe_value(row, "financial_risk_score")}

Financial Risk Level:
{safe_value(row, "financial_risk_level")}

Financial Risk Factors:
{safe_value(row, "financial_risk_factors")}


STATISTICAL ANOMALY

Statistical Anomaly Score:
{safe_value(row, "statistical_anomaly_score")}

Statistical Anomaly Level:
{safe_value(row, "statistical_anomaly_level")}

Statistical Anomaly Factors:
{safe_value(row, "statistical_anomaly_factors")}


ML ANOMALY DETECTION

Ensemble Risk Level:
{safe_value(row, "ensemble_risk_level")}

Combined Anomaly Score:
{safe_value(row, "combined_anomaly_score")}
"""

    return context


# ============================================================
# GENERATE LLM EXPLANATION
# ============================================================

def generate_llm_explanation(row):

    project_context = build_project_context(row)

    prompt = f"""
You are an AI assistant helping government auditors and investigators
analyze potentially risky public infrastructure projects.

Your task is to explain WHY this project was flagged by the AI risk system.

IMPORTANT RULES:

1. Do NOT invent facts.
2. Use ONLY the information provided.
3. Do NOT accuse anyone of corruption or fraud.
4. Use language such as:
   - potential anomaly
   - unusual financial pattern
   - requires verification
   - possible risk indicator
5. Clearly distinguish between:
   - Financial risk
   - Statistical anomaly
   - Machine learning anomaly
6. Explain the risk in simple language suitable for government auditors.
7. Recommend investigation steps.
8. Do not mention that you are an LLM.

PROJECT DATA:

{project_context}

Return the response in the following format:

EXECUTIVE SUMMARY:
A short 2-3 sentence explanation.

KEY RISK INDICATORS:
- indicator 1
- indicator 2
- indicator 3

WHY THE PROJECT WAS FLAGGED:
Explain the combination of risk signals.

FINANCIAL OBSERVATIONS:
Explain important financial patterns.

STATISTICAL OBSERVATIONS:
Explain statistical anomalies if present.

ML OBSERVATIONS:
Explain machine learning anomaly signals if present.

RECOMMENDED INVESTIGATION:
- investigation step 1
- investigation step 2
- investigation step 3

FINAL ASSESSMENT:
Provide a neutral professional assessment.
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# GENERATE EXPLANATIONS FOR HIGH-RISK PROJECTS
# ============================================================

print("\nGenerating LLM explanations...")

explanations = []


# Generate explanations only for projects
# that have some meaningful risk

risk_projects = df[
    df["final_ai_risk_score"] > 0
].copy()


print(
    "Projects selected for explanation:",
    len(risk_projects)
)


# ============================================================
# LIMIT INITIAL TEST
# ============================================================

TEST_LIMIT = 20


risk_projects = risk_projects.sort_values(

    "final_ai_risk_score",

    ascending=False

).head(TEST_LIMIT)


# ============================================================
# LLM LOOP
# ============================================================

for index, row in risk_projects.iterrows():

    print(
        f"\nGenerating explanation for: {row['work_id']}"
    )

    try:

        explanation = generate_llm_explanation(row)

    except Exception as e:

        print("Error:", e)

        explanation = "LLM explanation could not be generated."


    explanations.append({

        "work_id": row["work_id"],

        "final_ai_risk_score":
        row["final_ai_risk_score"],

        "final_ai_risk_level":
        row["final_ai_risk_level"],

        "risk_detection_confidence":
        safe_value(
            row,
            "risk_detection_confidence"
        ),

        "primary_risk_source":
        safe_value(
            row,
            "primary_risk_source"
        ),

        "llm_explanation":
        explanation

    })


# ============================================================
# CREATE OUTPUT DATAFRAME
# ============================================================

output_df = pd.DataFrame(explanations)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (

    "data/processed/"
    "llm_project_explanations.csv"

)


output_df.to_csv(

    output_path,

    index=False

)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)

print(
    "LLM EXPLANATION GENERATION COMPLETE"
)

print("=" * 60)


print("\nResults saved successfully!")

print(output_path)


print("\nSAMPLE EXPLANATION:\n")


if len(output_df) > 0:

    print(

        output_df.iloc[0][
            "llm_explanation"
        ]

    )