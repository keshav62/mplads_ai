from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_unified_risk_analysis_v2.csv"
)


class AnalyticsService:

    def __init__(self):

        self.data = pd.read_csv(
            DATA_FILE,
            low_memory=False
        )

        print(
            f"Analytics data loaded: "
            f"{len(self.data)} projects"
        )

    def get_overview(self):

        df = self.data.copy()

        # ---------------------------------
        # Total projects
        # ---------------------------------

        total_projects = len(df)

        # ---------------------------------
        # Risk distribution
        # ---------------------------------

        risk_counts = (
            df["final_ai_risk_level"]
            .astype(str)
            .str.upper()
            .value_counts()
            .to_dict()
        )

        low = int(
            risk_counts.get("LOW", 0)
        )

        medium = int(
            risk_counts.get("MEDIUM", 0)
        )

        high = int(
            risk_counts.get("HIGH", 0)
        )

        critical = int(
            risk_counts.get("CRITICAL", 0)
        )

        # ---------------------------------
        # Risk scores
        # ---------------------------------

        risk_scores = pd.to_numeric(
            df["final_ai_risk_score"],
            errors="coerce"
        )

        average_risk_score = risk_scores.mean()

        maximum_risk_score = risk_scores.max()

        # ---------------------------------
        # HIGH + CRITICAL
        # ---------------------------------

        high_critical_projects = (
            high + critical
        )

        # ---------------------------------
        # ML detected projects
        # ---------------------------------

        ml_detected_projects = (
            pd.to_numeric(
                df["ml_detected"],
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
            .sum()
        )

        # ---------------------------------
        # Multi-engine projects
        # ---------------------------------

        active_engines = pd.to_numeric(
            df["active_risk_engines"],
            errors="coerce"
        ).fillna(0)

        multi_engine_projects = (
            active_engines >= 2
        ).sum()

        # ---------------------------------
        # Return response
        # ---------------------------------

        return {

            "total_projects": total_projects,

            "risk_distribution": {
                "LOW": low,
                "MEDIUM": medium,
                "HIGH": high,
                "CRITICAL": critical
            },

            "average_risk_score": round(
                float(average_risk_score),
                2
            ),

            "maximum_risk_score": round(
                float(maximum_risk_score),
                2
            ),

            "high_critical_projects": (
                high_critical_projects
            ),

            "ml_detected_projects": int(
                ml_detected_projects
            ),

            "multi_engine_projects": int(
                multi_engine_projects
            )
        }

    def get_states(self):

        df = self.data.copy()

        result = (
            df.groupby("state")
            .agg(
                total_projects=("work_id", "count"),
                average_risk_score=(
                    "final_ai_risk_score",
                    "mean"
                )
            )
            .reset_index()
        )

        high = (
            df[df["final_ai_risk_level"] == "HIGH"]
            .groupby("state")
            .size()
            .rename("high_risk")
        )

        critical = (
            df[df["final_ai_risk_level"] == "CRITICAL"]
            .groupby("state")
            .size()
            .rename("critical_risk")
        )

        result = result.merge(
            high,
            on="state",
            how="left"
        )

        result = result.merge(
            critical,
            on="state",
            how="left"
        )

        result["high_risk"] = (
            result["high_risk"]
            .fillna(0)
            .astype(int)
        )

        result["critical_risk"] = (
            result["critical_risk"]
            .fillna(0)
            .astype(int)
        )

        result["average_risk_score"] = (
            result["average_risk_score"]
            .round(2)
        )

        result = result.sort_values(
            "total_projects",
            ascending=False
        )

        return {
            "total_states": len(result),
            "states": result.to_dict(
                orient="records"
            )
        }

    def get_categories(self):

        df = self.data.copy()

        result = (
            df.groupby("work_category")
            .agg(
                total_projects=("work_id", "count"),
                average_risk_score=(
                    "final_ai_risk_score",
                    "mean"
                )
            )
            .reset_index()
        )

        high = (
            df[df["final_ai_risk_level"] == "HIGH"]
            .groupby("work_category")
            .size()
            .rename("high_risk")
        )

        critical = (
            df[df["final_ai_risk_level"] == "CRITICAL"]
            .groupby("work_category")
            .size()
            .rename("critical_risk")
        )

        result = result.merge(
            high,
            on="work_category",
            how="left"
        )

        result = result.merge(
            critical,
            on="work_category",
            how="left"
        )

        result["high_risk"] = (
            result["high_risk"]
            .fillna(0)
            .astype(int)
        )

        result["critical_risk"] = (
            result["critical_risk"]
            .fillna(0)
            .astype(int)
        )

        result["average_risk_score"] = (
            result["average_risk_score"]
            .round(2)
        )

        result = result.sort_values(
            "total_projects",
            ascending=False
        )

        return {
            "total_categories": len(result),
            "categories": result.to_dict(
                orient="records"
            )
        }

    def get_constituencies(self):

        df = self.data.copy()

        result = (
            df.groupby(
                ["state", "constituency"]
            )
            .agg(
                total_projects=("work_id", "count"),
                average_risk_score=(
                    "final_ai_risk_score",
                    "mean"
                )
            )
            .reset_index()
        )

        high = (
            df[df["final_ai_risk_level"] == "HIGH"]
            .groupby(
                ["state", "constituency"]
            )
            .size()
            .rename("high_risk")
        )

        critical = (
            df[df["final_ai_risk_level"] == "CRITICAL"]
            .groupby(
                ["state", "constituency"]
            )
            .size()
            .rename("critical_risk")
        )

        result = result.merge(
            high,
            on=["state", "constituency"],
            how="left"
        )

        result = result.merge(
            critical,
            on=["state", "constituency"],
            how="left"
        )

        result["high_risk"] = (
            result["high_risk"]
            .fillna(0)
            .astype(int)
        )

        result["critical_risk"] = (
            result["critical_risk"]
            .fillna(0)
            .astype(int)
        )

        result["average_risk_score"] = (
            result["average_risk_score"]
            .round(2)
        )

        result = result.sort_values(
            "average_risk_score",
            ascending=False
        )

        return {
            "total_constituencies": len(result),
            "constituencies": result.to_dict(
                orient="records"
            )
        }