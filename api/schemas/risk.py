from pydantic import BaseModel
from typing import Dict, List


class RiskDistribution(BaseModel):

    LOW: int
    MEDIUM: int
    HIGH: int
    CRITICAL: int


class RiskSummaryResponse(BaseModel):

    total_projects: int
    risk_distribution: RiskDistribution
    average_risk_score: float
    high_critical_projects: int


class RiskDistributionResponse(BaseModel):

    labels: List[str]
    values: List[int]


class AnalyticsOverviewResponse(BaseModel):

    total_projects: int
    risk_distribution: RiskDistribution

    average_risk_score: float
    maximum_risk_score: float

    high_critical_projects: int
    ml_detected_projects: int
    multi_engine_projects: int