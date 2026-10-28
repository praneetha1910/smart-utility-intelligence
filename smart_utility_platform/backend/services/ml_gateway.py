"""Backend adapter to the ML layer; HTTP routes do not contain model logic."""
from __future__ import annotations
from typing import Any

from ml.inference import ModelRuntime


def get_model_runtime(models_dir: str | None = None) -> ModelRuntime:
    """Create the runtime used by the API. Routes share the app-scoped instance."""
    return ModelRuntime(models_dir=models_dir)


def predict(runtime: ModelRuntime, model_key: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Delegate validated records to the corresponding ML model artifact."""
    return runtime.predict(model_key, records)
