"""Request/response schemas for ML inference endpoints."""
from typing import Any
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Feature records for a fitted model; one dictionary per prediction row."""
    records: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)


class PredictionResponse(BaseModel):
    model: str
    model_version: str
    prediction_count: int
    predictions: list[Any]
    feature_columns: list[str]
    probabilities: list[list[float]] | None = None
    classes: list[Any] | None = None
