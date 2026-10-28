import json

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.inference import ModelNotReadyError, ModelRuntime, PredictionInputError


class SumEstimator:
    def predict(self, frame):
        return frame.sum(axis=1).to_numpy()


def test_model_runtime_reports_missing_artifact(tmp_path):
    runtime = ModelRuntime(tmp_path)
    status = runtime.status("electricity_forecasting")
    assert status["ready"] is False
    with pytest.raises(ModelNotReadyError):
        runtime.predict("electricity_forecasting", [{"a": 1}])


def test_model_runtime_loads_artifact_and_predicts(tmp_path):
    joblib.dump(SumEstimator(), tmp_path / "electricity_forecasting.joblib")
    (tmp_path / "electricity_forecasting.json").write_text(
        json.dumps({"version": "test", "feature_columns": ["a", "b"]}), encoding="utf-8"
    )

    runtime = ModelRuntime(tmp_path)
    result = runtime.predict("electricity_forecasting", [{"a": 2, "b": 3, "unused": 99}])

    assert result["predictions"] == [5]
    assert result["feature_columns"] == ["a", "b"]
    assert runtime.status("electricity_forecasting")["ready"] is True


def test_model_runtime_rejects_missing_feature(tmp_path):
    joblib.dump(SumEstimator(), tmp_path / "electricity_forecasting.joblib")
    (tmp_path / "electricity_forecasting.json").write_text(
        json.dumps({"feature_columns": ["a", "b"]}), encoding="utf-8"
    )
    with pytest.raises(PredictionInputError, match="Missing required feature"):
        ModelRuntime(tmp_path).predict("electricity_forecasting", [{"a": 1}])


def test_model_runtime_rejects_unknown_model(tmp_path):
    with pytest.raises(PredictionInputError, match="Unknown model"):
        ModelRuntime(tmp_path).status("unknown")
