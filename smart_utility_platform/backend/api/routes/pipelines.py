"""Dataset and model-pipeline lifecycle endpoints.

No processing or training starts on import/startup. Jobs begin only after explicit
POST requests, and the long-running scripts execute in worker threads via subprocess.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status

from backend.schemas.pipelines import PrepareRequest, TrainRequest
from backend.services.pipeline_service import (
    PipelineConflictError,
    PipelineManager,
    PipelineServiceError,
)

router = APIRouter(prefix="/api/v1", tags=["dataset pipelines"])


def _manager(request: Request) -> PipelineManager:
    return request.app.state.pipeline_manager


def _map_service_error(exc: PipelineServiceError) -> HTTPException:
    code = status.HTTP_409_CONFLICT if isinstance(exc, PipelineConflictError) else status.HTTP_400_BAD_REQUEST
    return HTTPException(status_code=code, detail=str(exc))


@router.get("/pipelines")
def list_pipelines(request: Request) -> dict:
    """List supported dataset pipelines and whether their inputs/results exist."""
    return _manager(request).list_pipelines()


@router.get("/datasets")
def dataset_inventory(request: Request) -> dict:
    """List raw source files and preparation/training readiness."""
    return _manager(request).inventory()


@router.get("/pipelines/{pipeline_key}/inspect")
def inspect_dataset(
    request: Request,
    pipeline_key: str,
    sample_rows: int = Query(default=200, ge=1, le=2000),
) -> dict:
    """Inspect sampled schemas without changing source data or starting training."""
    try:
        return _manager(request).inspect_dataset(pipeline_key, sample_rows=sample_rows)
    except PipelineServiceError as exc:
        raise _map_service_error(exc) from exc


@router.post("/pipelines/{pipeline_key}/prepare", status_code=status.HTTP_202_ACCEPTED)
def prepare_dataset(request: Request, pipeline_key: str, payload: PrepareRequest | None = None) -> dict:
    """Queue the appropriate dataset-cleaning/splitting script, but do not train."""
    try:
        job = _manager(request).submit(pipeline_key, "prepare", (payload or PrepareRequest()).model_dump())
        return {"message": "Dataset preparation job accepted.", "job": job}
    except PipelineServiceError as exc:
        raise _map_service_error(exc) from exc


@router.post("/pipelines/{pipeline_key}/train", status_code=status.HTTP_202_ACCEPTED)
def train_models(request: Request, pipeline_key: str, payload: TrainRequest | None = None) -> dict:
    """Queue model training only after prepared split files exist.

    This endpoint does not run automatically; it is provided for the later training phase.
    """
    try:
        job = _manager(request).submit(pipeline_key, "train", (payload or TrainRequest()).model_dump())
        return {"message": "Model training job explicitly accepted.", "job": job}
    except PipelineServiceError as exc:
        raise _map_service_error(exc) from exc


@router.get("/pipelines/{pipeline_key}/results")
def training_results(request: Request, pipeline_key: str) -> dict:
    """Fetch saved metrics/comparison table after a completed training run."""
    try:
        return _manager(request).get_training_results(pipeline_key)
    except PipelineServiceError as exc:
        raise _map_service_error(exc) from exc


@router.get("/jobs")
def recent_jobs(request: Request, limit: int = Query(default=20, ge=1, le=100)) -> dict:
    return {"jobs": _manager(request).recent_jobs(limit=limit)}


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict:
    job = _manager(request).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return {"job": job}
