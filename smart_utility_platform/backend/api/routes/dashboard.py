"""Dashboard summary endpoint for frontend integration health and readiness."""
from pathlib import Path
from fastapi import APIRouter, Request

from ml.data.loaders import discover_data_files

router = APIRouter(prefix="/api/v1", tags=["dashboard"])
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIRS = {
    "ampds2": PROJECT_ROOT / "data" / "raw" / "ampds2",
    "electricity_theft": PROJECT_ROOT / "data" / "raw" / "electricity_theft",
    "wdseventdb": PROJECT_ROOT / "data" / "raw" / "wdseventdb",
}


@router.get("/dashboard")
def dashboard_status(request: Request) -> dict:
    """Provide truthful dataset availability and model readiness for the UI."""
    datasets = {}
    for name, folder in DATASET_DIRS.items():
        found = discover_data_files(folder)
        datasets[name] = {
            "directory_exists": folder.is_dir(),
            "supported_files_found": len(found),
            "files": [file.name for file in found[:20]],
        }
    runtime = request.app.state.ml_runtime
    models = runtime.all_statuses()
    pipelines = request.app.state.pipeline_manager.list_pipelines()["pipelines"]
    pipeline_datasets = request.app.state.pipeline_manager.inventory()["datasets"]
    return {
        "api_status": "ok",
        "datasets": datasets,
        "pipeline_status": pipelines,
        "pipeline_datasets": pipeline_datasets,
        "models": models,
        "ready_model_count": sum(1 for model in models.values() if model["ready"]),
        "total_model_count": len(models),
        "note": "No prediction values are shown until fitted model artifacts are available.",
    }
