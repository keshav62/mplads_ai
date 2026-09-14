from fastapi import FastAPI

from api.routes.projects import router as projects_router
from api.routes.risk import router as risk_router
from api.routes.investigations import router as investigations_router
from api.routes.analytics import router as analytics_router

app = FastAPI(
    title="MPLADS AI Risk Intelligence API",
    description=(
        "AI-powered Government Project Risk "
        "Intelligence and Investigation System"
    ),
    version="1.0.0",
)


app.include_router(projects_router)
app.include_router(risk_router)
app.include_router(investigations_router)
app.include_router(analytics_router)


@app.get("/")
def root():

    return {
        "message": "MPLADS AI Risk Intelligence API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }