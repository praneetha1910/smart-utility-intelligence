"""FastAPI application entry point and dependency wiring."""
from fastapi import FastAPI

from backend.api.routes.dashboard import router as dashboard_router
from backend.api.routes.health import router as health_router
from backend.api.routes.ml import router as ml_router
from backend.api.routes.pipelines import router as pipelines_router
from backend.services.pipeline_service import PipelineManager
from backend.services.ml_gateway import get_model_runtime

app = FastAPI(
    title="Smart Energy, Water & Utility Intelligence API",
    version="0.2.0",
    description=(
        "Backend API for the Streamlit frontend. Inference requests are delegated "
        "to fitted ML artifacts stored under ml/models/."
    ),
)
app.state.ml_runtime = get_model_runtime()
app.state.pipeline_manager = PipelineManager()
app.include_router(health_router)
app.include_router(dashboard_router)
app.include_router(ml_router)
app.include_router(pipelines_router)


@app.get("/")
def root() -> dict[str, str]:
    """Basic API information."""
    return {
        "name": "Smart Energy, Water & Utility Intelligence API",
        "status": "ok",
        "docs": "/docs",
        "dashboard_status": "/api/v1/dashboard",
        "model_status": "/api/v1/models/status",
        "pipelines": "/api/v1/pipelines",
        "datasets": "/api/v1/datasets",
    }
