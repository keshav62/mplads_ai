from fastapi import APIRouter, HTTPException, Query

from api.services.project_service import ProjectService
from api.schemas.project import ProjectResponse


router = APIRouter(
    prefix="/projects",
    tags=["Projects"]
)


project_service = ProjectService()


@router.get("")
def get_projects(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    risk_level: str | None = None,
    state: str | None = None,
):
    return project_service.get_projects(
        limit=limit,
        offset=offset,
        risk_level=risk_level,
        state=state,
    )


@router.get(
    "/{work_id:path}",
    response_model=ProjectResponse
)
def get_project(work_id: str):

    project = project_service.get_project(work_id)

    if project is None:
        raise HTTPException(
            status_code=404,
            detail=f"Project '{work_id}' not found"
        )

    return project