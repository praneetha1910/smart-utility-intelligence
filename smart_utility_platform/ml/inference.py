"""Model-artifact loading and tabular inference shared by the backend.

Models are deliberately not fabricated. A model is available only when its fitted
artifact and metadata file have both been saved under ``ml/models/`` (or
``ML_MODELS_DIR``). Metadata must describe the feature columns used at training time.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


MODEL_SPECS: dict[str, dict[str, str]] = {
    "electricity_forecasting": {
        "label": "Electricity forecasting",
        "artifact": "electricity_forecasting.joblib",
        "metadata": "electricity_forecasting.json",
    },
    "electricity_theft_detection": {
        "label": "Electricity theft / anomaly detection",
        "artifact": "electricity_theft_detection.joblib",
        "metadata": "electricity_theft_detection.json",
    },
    "water_leakage_detection": {
        "label": "Water leakage detection",
        "artifact": "water_leakage_detection.joblib",
        "metadata": "water_leakage_detection.json",
    },
}


class ModelNotReadyError(RuntimeError):
    """Raised when a fitted model artifact has not been provided or loaded."""


class PredictionInputError(ValueError):
    """Raised when prediction input does not match a model's feature contract."""


class ModelRuntimeError(RuntimeError):
    """Raised when an artifact cannot safely perform inference."""


def _json_safe(value: Any) -> Any:
    """Convert NumPy/Pandas values to JSON-friendly Python values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) and not isinstance(value, (list, dict, tuple)):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


class ModelRuntime:
    """Loads fitted estimator artifacts and runs validated prediction requests."""

    def __init__(self, models_dir: str | Path | None = None) -> None:
        configured_dir = models_dir or os.getenv("ML_MODELS_DIR")
        self.models_dir = Path(configured_dir) if configured_dir else Path(__file__).resolve().parent / "models"
        self._cache: dict[str, tuple[int, Any, dict[str, Any]]] = {}

    def _paths(self, model_key: str) -> tuple[Path, Path]:
        if model_key not in MODEL_SPECS:
            raise PredictionInputError(f"Unknown model: {model_key}")
        spec = MODEL_SPECS[model_key]
        return self.models_dir / spec["artifact"], self.models_dir / spec["metadata"]

    def _load(self, model_key: str) -> tuple[Any, dict[str, Any]]:
        artifact_path, metadata_path = self._paths(model_key)
        missing = [str(path.name) for path in (artifact_path, metadata_path) if not path.is_file()]
        if missing:
            raise ModelNotReadyError(
                f"{MODEL_SPECS[model_key]['label']} is not trained/installed yet. "
                f"Missing file(s): {', '.join(missing)}"
            )

        modified_at = max(artifact_path.stat().st_mtime_ns, metadata_path.stat().st_mtime_ns)
        cached = self._cache.get(model_key)
        if cached and cached[0] == modified_at:
            return cached[1], cached[2]

        try:
            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            model = joblib.load(artifact_path)
        except Exception as exc:  # Bad artifact/metadata should be visible, not hidden.
            raise ModelRuntimeError(f"Could not load {model_key}: {exc}") from exc

        feature_columns = metadata.get("feature_columns")
        if not isinstance(feature_columns, list) or not feature_columns or not all(
            isinstance(column, str) and column for column in feature_columns
        ):
            raise ModelRuntimeError(
                f"Metadata for {model_key} must define a non-empty 'feature_columns' list."
            )
        if not callable(getattr(model, "predict", None)):
            raise ModelRuntimeError(f"Artifact for {model_key} does not provide a predict() method.")

        self._cache[model_key] = (modified_at, model, metadata)
        return model, metadata

    def status(self, model_key: str) -> dict[str, Any]:
        """Report readiness without inventing results if a model is absent."""
        if model_key not in MODEL_SPECS:
            raise PredictionInputError(f"Unknown model: {model_key}")
        spec = MODEL_SPECS[model_key]
        artifact_path, metadata_path = self._paths(model_key)
        base: dict[str, Any] = {
            "key": model_key,
            "name": spec["label"],
            "ready": False,
            "artifact_exists": artifact_path.is_file(),
            "metadata_exists": metadata_path.is_file(),
            "feature_columns": [],
            "version": None,
            "metrics": None,
            "message": "Train and save this model before requesting predictions.",
        }
        if not artifact_path.is_file() or not metadata_path.is_file():
            base["message"] = "Fitted model artifact and metadata are not both available yet."
            return base
        try:
            _, metadata = self._load(model_key)
            base.update(
                ready=True,
                feature_columns=metadata["feature_columns"],
                version=metadata.get("version", "unspecified"),
                metrics=metadata.get("metrics"),
                message="Model artifact and feature metadata loaded successfully.",
            )
        except (ModelRuntimeError, ModelNotReadyError) as exc:
            base["message"] = str(exc)
        return base

    def all_statuses(self) -> dict[str, dict[str, Any]]:
        return {key: self.status(key) for key in MODEL_SPECS}

    def predict(self, model_key: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Run an estimator against records matching its metadata feature columns.

        Each record is one inference row. Forecast rows must already contain the
        features expected by the fitted forecasting pipeline (lag/rolling features,
        calendar variables, etc.); this runtime never guesses them.
        """
        if not records:
            raise PredictionInputError("Provide at least one record to predict.")
        model, metadata = self._load(model_key)
        feature_columns: list[str] = metadata["feature_columns"]
        frame = pd.DataFrame.from_records(records)
        missing = [column for column in feature_columns if column not in frame.columns]
        if missing:
            raise PredictionInputError(
                f"Missing required feature(s): {', '.join(missing)}. "
                f"Required features: {feature_columns}"
            )

        features = frame.loc[:, feature_columns]
        try:
            raw_predictions = model.predict(features)
        except Exception as exc:
            raise PredictionInputError(f"Model could not process the supplied features: {exc}") from exc

        predictions = _json_safe(np.asarray(raw_predictions).reshape(-1))
        if len(predictions) != len(records):
            raise ModelRuntimeError(
                f"Model returned {len(predictions)} prediction(s) for {len(records)} input record(s)."
            )

        response: dict[str, Any] = {
            "model": model_key,
            "model_version": metadata.get("version", "unspecified"),
            "prediction_count": len(predictions),
            "predictions": predictions,
            "feature_columns": feature_columns,
        }
        predict_proba = getattr(model, "predict_proba", None)
        if callable(predict_proba):
            try:
                probabilities = np.asarray(predict_proba(features))
                classes = getattr(model, "classes_", None)
                response["probabilities"] = _json_safe(probabilities)
                if classes is not None:
                    response["classes"] = _json_safe(np.asarray(classes))
            except Exception:
                # Probability output is optional; the primary prediction remains valid.
                response["probabilities"] = None
        return response
