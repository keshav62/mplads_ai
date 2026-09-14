from fastapi import APIRouter

from api.services.analytics_service import AnalyticsService
from api.schemas.risk import AnalyticsOverviewResponse


router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"]
)


analytics_service = AnalyticsService()


@router.get(
    "/overview",
    response_model=AnalyticsOverviewResponse
)
def get_overview():

    return analytics_service.get_overview()


@router.get("/states")
def get_states():

    return analytics_service.get_states()


@router.get("/categories")
def get_categories():

    return analytics_service.get_categories()


@router.get("/constituencies")
def get_constituencies():

    return analytics_service.get_constituencies()