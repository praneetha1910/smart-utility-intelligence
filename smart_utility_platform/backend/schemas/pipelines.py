"""Request schemas for the dataset-pipeline orchestration API."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class PrepareRequest(BaseModel):
    """Options for preparing a dataset. Does not train a model."""
    target_meter: str = Field(default="WHE", pattern=r"^[A-Za-z0-9_]{1,20}$")
    horizon_hours: int = Field(default=1, ge=1, le=168)
    submeters: list[str] = Field(default_factory=list, max_length=16)
    csv_filename: str | None = Field(default=None, max_length=300)
    keep_all_zero_customers: bool = False
    disable_winsorization: bool = False


class TrainRequest(BaseModel):
    """Options for an explicitly requested model-training run."""
    gbm: Literal["auto", "lightgbm", "xgboost", "hist"] = "auto"
    n_trees: int = Field(default=200, ge=10, le=2000)
    n_jobs: int = Field(default=1, ge=1, le=8)
    skip_permutation_importance: bool = True
    no_refit: bool = False
    isolation_forest: bool = False
    gbm_rounds: int = Field(default=100, ge=10, le=2000)
    train_stride: int = Field(default=1, ge=1, le=1000)
    cv_max_folds: int = Field(default=4, ge=2, le=20)
