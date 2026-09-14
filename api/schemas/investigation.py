from pydantic import BaseModel
from typing import Optional


class InvestigationResponse(BaseModel):

    investigation_rank: Optional[int] = None

    work_id: str
    state: Optional[str] = None
    constituency: Optional[str] = None

    final_ai_risk_score: Optional[float] = None
    final_ai_risk_level: Optional[str] = None

    risk_detection_confidence: Optional[str] = None

    active_risk_engines: Optional[int] = None
    available_risk_engines: Optional[int] = None

    primary_risk_source: Optional[str] = None
    detecting_engines: Optional[str] = None

    investigation_priority_score: Optional[float] = None
    investigation_priority_category: Optional[str] = None


class InvestigationReportResponse(BaseModel):

    investigation_rank: Optional[int] = None

    work_id: str
    state: Optional[str] = None
    constituency: Optional[str] = None

    final_ai_risk_score: Optional[float] = None
    final_ai_risk_level: Optional[str] = None

    risk_detection_confidence: Optional[str] = None

    active_risk_engines: Optional[int] = None
    available_risk_engines: Optional[int] = None

    primary_risk_source: Optional[str] = None

    investigation_priority_score: Optional[float] = None
    investigation_priority_category: Optional[str] = None

    report_status: Optional[str] = None
    report_file: Optional[str] = None
    grounded_llm_investigation_report: Optional[str] = None