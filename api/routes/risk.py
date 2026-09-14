from fastapi import APIRouter

from api.services.risk_service import RiskService
from api.schemas.risk import (
    RiskSummaryResponse,
    RiskDistributionResponse
)


router = APIRouter(
    prefix="/risk",
    tags=["Risk Analysis"]
)


risk_service = RiskService()


@router.get(
    "/summary",
    response_model=RiskSummaryResponse
)
def get_risk_summary():

    return risk_service.get_summary()


@router.get(
    "/distribution",
    response_model=RiskDistributionResponse
)
def get_risk_distribution():

    return risk_service.get_distribution()