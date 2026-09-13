import pandas as pd
from fastapi import FastAPI, HTTPException
import numpy as np
from fastapi import Query
from fastapi import HTTPException


app = FastAPI(
    title="MPLADS AI Risk Monitoring API",
    description="AI-powered project risk and anomaly detection system",
    version="1.0.0"
)


risk_df = pd.read_csv(
    "data/processed/risk_scored_projects.csv"
)


@app.get("/")
def home():
    return {
        "message": "MPLADS AI Risk Monitoring API is running successfully"
    }


def clean_json_data(data):

    if isinstance(data, pd.DataFrame):

        data = data.replace(
            [np.nan, np.inf, -np.inf],
            None
        )

        return data.to_dict(
            orient="records"
        )

    return data


@app.get("/dashboard-summary")
def get_dashboard_summary():

    total_projects = len(risk_df)

    high_risk = len(
        risk_df[
            risk_df["risk_level"] == "HIGH"
        ]
    )

    medium_risk = len(
        risk_df[
            risk_df["risk_level"] == "MEDIUM"
        ]
    )

    low_risk = len(
        risk_df[
            risk_df["risk_level"] == "LOW"
        ]
    )

    total_anomalies = len(
        risk_df[
            risk_df["is_anomaly"] == 1
        ]
    )

    high_risk_percentage = round(
        (high_risk / total_projects) * 100,
        2
    ) if total_projects > 0 else 0

    return {
        "total_projects": total_projects,
        "high_risk_projects": high_risk,
        "medium_risk_projects": medium_risk,
        "low_risk_projects": low_risk,
        "total_anomalies": total_anomalies,
        "high_risk_percentage": high_risk_percentage
    }


@app.get("/high-risk-projects")
def get_high_risk_projects():

    high_risk_df = risk_df[
        risk_df["risk_level"] == "HIGH"
    ].copy()

    columns = [
        "work_id",
        "state",
        "constituency",
        "recommended_amount",
        "sanction_amount",
        "total_expenditure",
        "risk_score",
        "risk_level",
        "risk_reasons",
        "anomaly_score"
    ]

    available_columns = [
        col for col in columns
        if col in high_risk_df.columns
    ]

    high_risk_projects = (
        high_risk_df[available_columns]
        .sort_values(
            by="risk_score",
            ascending=False
        )
        .replace({
            np.nan: None,
            np.inf: None,
            -np.inf: None
        })
        .to_dict(orient="records")
    )

    return {
        "total_high_risk_projects": len(high_risk_projects),
        "projects": high_risk_projects
    }


from fastapi import Query, HTTPException


@app.get("/projects")
def get_all_projects(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    state: str | None = None,
    risk_level: str | None = None,
    search: str | None = None,
    sort_by: str = "risk_score",
    order: str = "desc"
):

    filtered_df = risk_df.copy()

    # -----------------------------
    # FILTER BY STATE
    # -----------------------------
    if state:
        filtered_df = filtered_df[
            filtered_df["state"].str.contains(
                state,
                case=False,
                na=False
            )
        ]

    # -----------------------------
    # FILTER BY RISK LEVEL
    # -----------------------------
    if risk_level:
        filtered_df = filtered_df[
            filtered_df["risk_level"].str.upper()
            == risk_level.upper()
        ]

    # -----------------------------
    # SEARCH
    # -----------------------------
    if search:

        filtered_df = filtered_df[
            filtered_df["work_id"].str.contains(
                search,
                case=False,
                na=False
            )
            |
            filtered_df["constituency"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    # -----------------------------
    # SORTING
    # -----------------------------

    allowed_sort_columns = [
        "risk_score",
        "recommended_amount",
        "total_expenditure",
        "anomaly_score"
    ]

    # Validate sort column
    if sort_by not in allowed_sort_columns:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid sort_by value",
                "allowed_values": allowed_sort_columns
            }
        )

    # Validate order
    if order.lower() not in ["asc", "desc"]:
        raise HTTPException(
            status_code=400,
            detail="Order must be 'asc' or 'desc'"
        )

    # Sort dataframe
    ascending = order.lower() == "asc"

    filtered_df = filtered_df.sort_values(
        by=sort_by,
        ascending=ascending,
        na_position="last"
    )

    # -----------------------------
    # PAGINATION
    # -----------------------------

    total_projects = len(filtered_df)

    start = (page - 1) * limit
    end = start + limit

    paginated_df = filtered_df.iloc[start:end].copy()

    # -----------------------------
    # SELECT COLUMNS
    # -----------------------------

    columns = [
        "work_id",
        "state",
        "constituency",
        "work_category",
        "recommended_amount",
        "sanction_amount",
        "total_expenditure",
        "risk_score",
        "risk_level",
        "risk_reasons",
        "anomaly_score"
    ]

    available_columns = [
        col for col in columns
        if col in paginated_df.columns
    ]

    paginated_df = paginated_df[available_columns]

    # -----------------------------
    # HANDLE NaN VALUES
    # -----------------------------

    paginated_df = paginated_df.astype(object).where(
        paginated_df.notna(),
        None
    )

    projects = paginated_df.to_dict(
        orient="records"
    )

    # -----------------------------
    # RESPONSE
    # -----------------------------

    return {
        "page": page,
        "limit": limit,
        "total_projects": total_projects,
        "total_pages": (
            total_projects + limit - 1
        ) // limit,

        "sorting": {
            "sort_by": sort_by,
            "order": order.lower()
        },

        "projects": projects
    }


@app.get("/project")
def get_project_details(work_id: str):

    project = risk_df[
        risk_df["work_id"] == work_id
    ]

    if project.empty:
        raise HTTPException(
            status_code=404,
            detail="Project not found"
        )

    project_data = project.iloc[0].copy()

    # Convert NaN values to None
    project_data = project_data.where(
        project_data.notna(),
        None
    )

    return project_data.to_dict()



@app.get("/state-risk-summary")
def get_state_risk_summary():

    state_summary = (
        risk_df
        .groupby("state")
        .agg(
            total_projects=("work_id", "count"),
            average_risk_score=("risk_score", "mean")
        )
        .reset_index()
    )

    # Count projects by risk level
    risk_counts = (
        pd.crosstab(
            risk_df["state"],
            risk_df["risk_level"]
        )
        .reset_index()
    )

    # Merge both results
    state_summary = state_summary.merge(
        risk_counts,
        on="state",
        how="left"
    )

    # Make sure all risk-level columns exist
    for column in ["HIGH", "MEDIUM", "LOW"]:

        if column not in state_summary.columns:
            state_summary[column] = 0

    # Rename columns
    state_summary = state_summary.rename(
        columns={
            "HIGH": "high_risk",
            "MEDIUM": "medium_risk",
            "LOW": "low_risk"
        }
    )

    # Round average risk score
    state_summary["average_risk_score"] = (
        state_summary["average_risk_score"]
        .round(2)
    )

    # Sort states by high-risk projects
    state_summary = state_summary.sort_values(
        by="high_risk",
        ascending=False
    )

    # Convert NaN to None for JSON
    state_summary = state_summary.where(
        state_summary.notna(),
        None
    )

    return {
        "total_states": len(state_summary),
        "states": state_summary.to_dict(
            orient="records"
        )
    }


@app.get("/analytics/project-lifecycle")
def get_project_lifecycle():

    total_projects = len(risk_df)

    sanctioned = int(
        risk_df["has_sanction"].sum()
    )

    completed = int(
        risk_df["has_completion"].sum()
    )

    expenditure = int(
        risk_df["has_expenditure"].sum()
    )

    return {
        "total_projects": total_projects,

        "sanctioned_projects": sanctioned,

        "completed_projects": completed,

        "projects_with_expenditure": expenditure,

        "sanction_rate": round(
            (sanctioned / total_projects) * 100,
            2
        ) if total_projects > 0 else 0,

        "completion_rate": round(
            (completed / total_projects) * 100,
            2
        ) if total_projects > 0 else 0,

        "expenditure_rate": round(
            (expenditure / total_projects) * 100,
            2
        ) if total_projects > 0 else 0
    }


@app.get("/analytics/risk-distribution")
def get_risk_distribution():

    distribution = (
        risk_df["risk_level"]
        .value_counts()
        .reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
    )

    return {
        "total_projects": int(len(risk_df)),
        "risk_distribution": {
            "LOW": int(distribution["LOW"]),
            "MEDIUM": int(distribution["MEDIUM"]),
            "HIGH": int(distribution["HIGH"])
        }
    }

@app.get("/analytics/top-risk-states")
def get_top_risk_states():

    state_summary = (
        risk_df
        .groupby("state")
        .agg(
            total_projects=("work_id", "count"),
            high_risk_projects=(
                "risk_level",
                lambda x: (x == "HIGH").sum()
            ),
            medium_risk_projects=(
                "risk_level",
                lambda x: (x == "MEDIUM").sum()
            ),
            average_risk_score=(
                "risk_score",
                "mean"
            )
        )
        .reset_index()
    )

    state_summary = state_summary.sort_values(
        by=[
            "high_risk_projects",
            "medium_risk_projects",
            "average_risk_score"
        ],
        ascending=False
    )

    # Convert NaN values to None for JSON
    state_summary = state_summary.replace(
        [np.nan, np.inf, -np.inf],
        None
    )

    return {
        "total_states": int(len(state_summary)),
        "states": state_summary.to_dict(
            orient="records"
        )
    }

@app.get("/analytics/financial-summary")
def get_financial_summary():

    total_recommended = (
        risk_df["recommended_amount"]
        .sum()
    )

    total_sanctioned = (
        risk_df["sanction_amount"]
        .sum()
    )

    total_expenditure = (
        risk_df["total_expenditure"]
        .sum()
    )

    average_recommended = (
        risk_df["recommended_amount"]
        .mean()
    )

    average_expenditure = (
        risk_df["total_expenditure"]
        .mean()
    )

    return {
        "total_recommended_amount": (
            float(total_recommended)
            if pd.notna(total_recommended)
            else 0
        ),

        "total_sanctioned_amount": (
            float(total_sanctioned)
            if pd.notna(total_sanctioned)
            else 0
        ),

        "total_expenditure": (
            float(total_expenditure)
            if pd.notna(total_expenditure)
            else 0
        ),

        "average_recommended_amount": (
            float(average_recommended)
            if pd.notna(average_recommended)
            else 0
        ),

        "average_expenditure": (
            float(average_expenditure)
            if pd.notna(average_expenditure)
            else 0
        ),

        "projects_with_expenditure": int(
            risk_df["total_expenditure"]
            .notna()
            .sum()
        )
    }



def create_alert_dataframe():

    # Select HIGH and MEDIUM risk projects
    alert_df = risk_df[
        risk_df["risk_level"].isin(
            ["HIGH", "MEDIUM"]
        )
    ].copy()

    # Create unique alert ID
    alert_df["alert_id"] = (
        "ALERT-" + alert_df.index.astype(str)
    )

    # Create alert severity
    alert_df["alert_severity"] = alert_df[
        "risk_level"
    ].map({
        "HIGH": "CRITICAL",
        "MEDIUM": "WARNING"
    })

    # Generate alert messages
    def generate_alert_message(row):

        if row["risk_level"] == "HIGH":
            return (
                "High-risk project detected. "
                "Immediate review is recommended."
            )

        return (
            "Medium-risk project detected. "
            "Further monitoring is recommended."
        )

    alert_df["alert_message"] = alert_df.apply(
        generate_alert_message,
        axis=1
    )

    return alert_df 

@app.get("/alerts/summary")
def get_alert_summary():

    # -----------------------------------
    # SELECT ALERT PROJECTS
    # -----------------------------------

    alert_df = risk_df[
        risk_df["risk_level"].isin(
            ["HIGH", "MEDIUM"]
        )
    ].copy()


    # -----------------------------------
    # CREATE ALERT SEVERITY
    # -----------------------------------

    alert_df["alert_severity"] = alert_df[
        "risk_level"
    ].map({
        "HIGH": "CRITICAL",
        "MEDIUM": "WARNING"
    })


    # -----------------------------------
    # TOTAL ALERT COUNTS
    # -----------------------------------

    total_alerts = len(alert_df)

    critical_alerts = int(
        (alert_df["alert_severity"] == "CRITICAL").sum()
    )

    warning_alerts = int(
        (alert_df["alert_severity"] == "WARNING").sum()
    )


    # -----------------------------------
    # STATES WITH MOST ALERTS
    # -----------------------------------

    state_alerts = (
        alert_df
        .groupby("state")
        .agg(
            total_alerts=("work_id", "count"),

            critical_alerts=(
                "alert_severity",
                lambda x: (x == "CRITICAL").sum()
            ),

            warning_alerts=(
                "alert_severity",
                lambda x: (x == "WARNING").sum()
            ),

            average_risk_score=(
                "risk_score",
                "mean"
            )
        )
        .reset_index()
        .sort_values(
            by=["total_alerts", "critical_alerts"],
            ascending=False
        )
        .head(10)
    )


    # -----------------------------------
    # TOP RISK PROJECTS
    # -----------------------------------

    top_projects_columns = [
        "work_id",
        "state",
        "constituency",
        "risk_score",
        "risk_level",
        "risk_reasons",
        "anomaly_score"
    ]


    available_columns = [
        col for col in top_projects_columns
        if col in alert_df.columns
    ]


    top_risk_projects = (
        alert_df
        .sort_values(
            by="risk_score",
            ascending=False
        )
        [available_columns]
        .head(10)
    )


    # -----------------------------------
    # CLEAN NaN VALUES FOR JSON
    # -----------------------------------

    state_alerts = state_alerts.replace(
        [np.nan, np.inf, -np.inf],
        None
    )

    top_risk_projects = top_risk_projects.replace(
        [np.nan, np.inf, -np.inf],
        None
    )


    # -----------------------------------
    # RETURN RESPONSE
    # -----------------------------------

    return {

        "alert_summary": {

            "total_alerts": int(total_alerts),

            "critical_alerts": int(critical_alerts),

            "warning_alerts": int(warning_alerts)

        },


        "states_with_most_alerts":

            state_alerts.to_dict(
                orient="records"
            ),


        "top_risk_projects":

            top_risk_projects.to_dict(
                orient="records"
            )
    }

@app.get("/alerts/{alert_id}")
def get_alert_details(alert_id: str):

    alert_df = create_alert_dataframe()

    alert = alert_df[
        alert_df["alert_id"] == alert_id
    ]

    if alert.empty:

        raise HTTPException(
            status_code=404,
            detail="Alert not found"
        )

    alert_details = (
        alert
        .replace([np.nan, np.inf, -np.inf], None)
        .iloc[0]
        .to_dict()
    )

    return alert_details


@app.get("/alerts")
def get_alerts(
    severity: str = Query(
        None,
        description="Filter alerts by severity: CRITICAL or WARNING"
    ),
    state: str = Query(
        None,
        description="Filter alerts by state"
    ),
    limit: int = Query(
        50,
        ge=1,
        le=500,
        description="Maximum number of alerts to return"
    )
):

    # Create alerts
    alert_df = create_alert_dataframe()


    # -----------------------------------
    # FILTER BY SEVERITY
    # -----------------------------------

    if severity:

        severity = severity.upper()

        alert_df = alert_df[
            alert_df["alert_severity"] == severity
        ]


    # -----------------------------------
    # FILTER BY STATE
    # -----------------------------------

    if state:

        alert_df = alert_df[
            alert_df["state"]
            .str.lower()
            == state.lower()
        ]


    # -----------------------------------
    # SORT BY RISK SCORE
    # -----------------------------------

    alert_df = alert_df.sort_values(
        by="risk_score",
        ascending=False
    )


    # -----------------------------------
    # TOTAL BEFORE LIMIT
    # -----------------------------------

    total_filtered_alerts = len(alert_df)


    # -----------------------------------
    # APPLY LIMIT
    # -----------------------------------

    alert_df = alert_df.head(limit)


    # -----------------------------------
    # SELECT RESPONSE COLUMNS
    # -----------------------------------

    columns = [
        "alert_id",
        "work_id",
        "state",
        "constituency",
        "recommended_amount",
        "sanction_amount",
        "total_expenditure",
        "risk_score",
        "risk_level",
        "alert_severity",
        "alert_message",
        "risk_reasons",
        "anomaly_score"
    ]


    available_columns = [
        col for col in columns
        if col in alert_df.columns
    ]


    alert_df = alert_df[
        available_columns
    ]


    # -----------------------------------
    # CLEAN JSON VALUES
    # -----------------------------------

    alert_df = alert_df.replace(
        [np.nan, np.inf, -np.inf],
        None
    )


    alerts = alert_df.to_dict(
        orient="records"
    )


    return {

        "filters": {
            "severity": severity,
            "state": state,
            "limit": limit
        },

        "total_filtered_alerts": total_filtered_alerts,

        "returned_alerts": len(alerts),

        "alerts": alerts
    }