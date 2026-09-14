from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

QUEUE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "investigation_queue.csv"
)

REPORT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "grounded_llm_investigation_reports_v4.csv"
)


class InvestigationService:

    def __init__(self):

        self.queue = pd.read_csv(
            QUEUE_FILE,
            low_memory=False
        )

        self.reports = pd.read_csv(
            REPORT_FILE,
            low_memory=False
        )

        print(
            f"Investigation queue loaded: "
            f"{len(self.queue)} projects"
        )

        print(
            f"Investigation reports loaded: "
            f"{len(self.reports)} reports"
        )

    def _clean_records(self, df):

        # Replace infinity values
        df = df.replace(
            [float("inf"), float("-inf")],
            None
        )

        # Convert NaN to None
        df = df.astype(object).where(
            pd.notna(df),
            None
        )

        return df.to_dict(orient="records")

    def get_investigations(
        self,
        limit=20,
        offset=0,
        risk_level=None,
        priority_category=None,
        state=None,
    ):

        df = self.queue.copy()

        # Risk level filter
        if risk_level:

            df = df[
                df["final_ai_risk_level"]
                .astype(str)
                .str.upper()
                == risk_level.upper()
            ]

        # Priority category filter
        if priority_category:

            df = df[
                df["investigation_priority_category"]
                .astype(str)
                .str.lower()
                .str.contains(
                    priority_category.lower(),
                    na=False
                )
            ]

        # State filter
        if state:

            df = df[
                df["state"]
                .astype(str)
                .str.lower()
                .str.contains(
                    state.lower(),
                    na=False
                )
            ]

        total = len(df)

        # Pagination
        df = df.iloc[
            offset:offset + limit
        ]

        records = self._clean_records(df)

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "investigations": records,
        }

    def get_investigation(self, work_id):

        df = self.queue[
            self.queue["work_id"].astype(str)
            == str(work_id)
        ]

        if df.empty:
            return None

        record = self._clean_records(df)

        return record[0]

    def get_report(self, work_id):

        df = self.reports[
            self.reports["work_id"].astype(str)
            == str(work_id)
        ]

        if df.empty:
            return None

        record = self._clean_records(df)

        return record[0]