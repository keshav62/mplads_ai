from fastapi import APIRouter, HTTPException, Query

from api.services.investigation_service import InvestigationService
from api.schemas.investigation import (
    InvestigationResponse,
    InvestigationReportResponse
)


router = APIRouter(
    prefix="/investigations",
    tags=["Investigations"]
)

investigation_service = InvestigationService()


@router.get("")
def get_investigations(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    risk_level: str | None = None,
    priority_category: str | None = None,
    state: str | None = None,
):
    return investigation_service.get_investigations(
        limit=limit,
        offset=offset,
        risk_level=risk_level,
        priority_category=priority_category,
        state=state,
    )


@router.get(
    "/{work_id:path}/report",
    response_model=InvestigationReportResponse
)
def get_investigation_report(work_id: str):

    report = investigation_service.get_report(work_id)

    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"Investigation report '{work_id}' not found"
        )

    return report


@router.get(
    "/{work_id:path}",
    response_model=InvestigationResponse
)
def get_investigation(work_id: str):

    investigation = investigation_service.get_investigation(work_id)

    if investigation is None:
        raise HTTPException(
            status_code=404,
            detail=f"Investigation '{work_id}' not found"
        )

    return investigation