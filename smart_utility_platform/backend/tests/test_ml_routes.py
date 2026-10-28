from fastapi.testclient import TestClient

from backend.main import app
from ml.inference import ModelRuntime

client = TestClient(app)


class SumEstimator:
    def predict(self, frame):
        return frame.sum(axis=1).to_numpy()


def test_dashboard_reports_backend_and_model_readiness(tmp_path):
    original = app.state.ml_runtime
    app.state.ml_runtime = ModelRuntime(tmp_path)
    try:
        response = client.get("/api/v1/dashboard")
        assert response.status_code == 200
        payload = response.json()
        assert payload["api_status"] == "ok"
        assert payload["ready_model_count"] == 0
        assert payload["total_model_count"] == 3
        assert payload["models"]["electricity_forecasting"]["ready"] is False
        assert {item["key"] for item in payload["pipeline_status"]} == {
            "ampds2", "electricity_theft", "wdseventdb"
        }
    finally:
        app.state.ml_runtime = original


def test_prediction_route_returns_503_when_model_is_not_installed(tmp_path):
    original = app.state.ml_runtime
    app.state.ml_runtime = ModelRuntime(tmp_path)
    try:
        response = client.post("/api/v1/forecast", json={"records": [{"a": 1}]})
        assert response.status_code == 503
        assert "not trained/installed yet" in response.json()["detail"]
    finally:
        app.state.ml_runtime = original


def test_prediction_route_validates_nonempty_records():
    response = client.post("/api/v1/forecast", json={"records": []})
    assert response.status_code == 422


def test_ml_model_status_endpoint_lists_three_models(tmp_path):
    original = app.state.ml_runtime
    app.state.ml_runtime = ModelRuntime(tmp_path)
    try:
        response = client.get("/api/v1/models/status")
        assert response.status_code == 200
        assert len(response.json()["models"]) == 3
    finally:
        app.state.ml_runtime = original


def test_forecast_endpoint_delegates_to_installed_ml_artifact(tmp_path):
    import json
    import joblib

    joblib.dump(SumEstimator(), tmp_path / "electricity_forecasting.joblib")
    (tmp_path / "electricity_forecasting.json").write_text(
        json.dumps({"version": "integration-test", "feature_columns": ["a", "b"]}),
        encoding="utf-8",
    )
    original = app.state.ml_runtime
    app.state.ml_runtime = ModelRuntime(tmp_path)
    try:
        response = client.post(
            "/api/v1/forecast",
            json={"records": [{"a": 2, "b": 5, "unneeded": 100}]},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["predictions"] == [7]
        assert payload["model_version"] == "integration-test"
    finally:
        app.state.ml_runtime = original
