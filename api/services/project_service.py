from pathlib import Path
import pandas as pd


# Project root: E:\mplads_ai
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_unified_risk_analysis_v2.csv"
)


class ProjectService:

    def __init__(self):
        self.data = pd.read_csv(
            DATA_FILE,
            low_memory=False
        )

        print(f"Loaded {len(self.data)} projects")
        print(f"Columns: {len(self.data.columns)}")

    def _clean_records(self, df):
        """
        Convert Pandas/NumPy values into JSON-safe Python values.
        NaN, +inf and -inf become None.
        """

        # Replace infinite values
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

    def get_projects(
        self,
        limit: int = 20,
        offset: int = 0,
        risk_level: str | None = None,
        state: str | None = None,
    ):

        df = self.data

        # Filter by risk level
        if risk_level:
            df = df[
                df["final_ai_risk_level"]
                .astype(str)
                .str.upper()
                == risk_level.upper()
            ]

        # Filter by state
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
            "projects": records,
        }

    def get_project(self, work_id: str):

        df = self.data[
            self.data["work_id"].astype(str)
            == str(work_id)
        ]

        if df.empty:
            return None

        records = self._clean_records(df)

        return records[0]