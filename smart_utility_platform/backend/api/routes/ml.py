"""HTTP endpoints that delegate inference to the ML runtime."""
from fastapi import APIRouter, HTTPException, Request, status

from backend.schemas.predictions import PredictionRequest, PredictionResponse
from ml.inference import ModelNotReadyError, ModelRuntimeError, PredictionInputError

router = APIRouter(prefix="/api/v1", tags=["machine learning"])


def _runtime(request: Request):
    return request.app.state.ml_runtime


def _run_prediction(request: Request, model_key: str, payload: PredictionRequest) -> dict:
    try:
        return _runtime(request).predict(model_key, payload.records)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except PredictionInputError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except ModelRuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.get("/models/status")
def model_status(request: Request) -> dict:
    """Report readiness and expected feature columns for the three model modules."""
    return {"models": _runtime(request).all_statuses()}


@router.post("/forecast", response_model=PredictionResponse)
def forecast(request: Request, payload: PredictionRequest) -> dict:
    """Predict consumption for each provided feature row using a fitted model."""
    return _run_prediction(request, "electricity_forecasting", payload)


@router.post("/theft/predict", response_model=PredictionResponse)
def detect_theft(request: Request, payload: PredictionRequest) -> dict:
    """Classify/analyze each provided meter feature row using a fitted model."""
    return _run_prediction(request, "electricity_theft_detection", payload)


@router.post("/water-leakage/predict", response_model=PredictionResponse)
def detect_water_leakage(request: Request, payload: PredictionRequest) -> dict:
    """Classify each provided water-system feature row using a fitted model."""
    return _run_prediction(request, "water_leakage_detection", payload)
