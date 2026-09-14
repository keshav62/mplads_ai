from pydantic import BaseModel
from typing import Optional


class ProjectResponse(BaseModel):

    work_id: str
    state: Optional[str] = None
    constituency: Optional[str] = None
    work_category: Optional[str] = None
    work_description: Optional[str] = None

    recommended_amount: Optional[float] = None
    sanction_amount: Optional[float] = None
    total_expenditure: Optional[float] = None

    payment_count: Optional[float] = None
    unique_vendors: Optional[float] = None
    payments_per_vendor: Optional[float] = None

    sanction_delay_days: Optional[float] = None
    completion_duration_days: Optional[float] = None

    expenditure_ratio: Optional[float] = None
    sanction_ratio: Optional[float] = None

    ml_anomaly_score: Optional[float] = None
    ensemble_is_anomaly: Optional[int] = None
    both_models_anomaly: Optional[int] = None
    model_agreement: Optional[str] = None
    primary_ml_source: Optional[str] = None
    ensemble_risk_level: Optional[str] = None

    rule_risk_score: Optional[float] = None
    financial_risk_score: Optional[float] = None
    statistical_anomaly_score: Optional[float] = None

    active_risk_engines: Optional[int] = None
    available_risk_engines: Optional[int] = None

    final_ai_risk_score: Optional[float] = None
    final_ai_risk_level: Optional[str] = None
    risk_detection_confidence: Optional[str] = None
    primary_risk_source: Optional[str] = None

    combined_risk_factor_count: Optional[int] = None
    detecting_engines: Optional[str] = None