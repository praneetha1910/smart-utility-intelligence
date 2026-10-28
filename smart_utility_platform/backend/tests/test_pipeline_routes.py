"""Tests for pipeline status, input validation, and explicit job execution."""
from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.pipeline_service import PipelineManager

client = TestClient(app)


def make_fake_project(root: Path) -> PipelineManager:
    pipeline_dir = root / "ml" / "pipelines"
    raw = root / "data" / "raw" / "ampds2"
    pipeline_dir.mkdir(parents=True)
    raw.mkdir(parents=True)
    (raw / "Electricity_WHE.csv").write_text("unix_ts,P\n1,5\n", encoding="utf-8")
    (raw / "Climate_HourlyWeather.csv").write_text("Date/Time,Temp (C)\n2012-01-01 00:00,3\n", encoding="utf-8")
    # A lightweight stand-in verifies the API calls the fixed CLI contract. No training is run.
    (pipeline_dir / "ampds2_pipeline.py").write_text(
        """import argparse\nfrom pathlib import Path\np=argparse.ArgumentParser()\np.add_argument('--data-dir')\np.add_argument('--out-dir')\np.add_argument('--target-meter')\np.add_argument('--horizon')\np.add_argument('--submeters', nargs='*', default=[])\na=p.parse_args()\nout=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)\nfor name in ['ampds2_train.csv','ampds2_val.csv','ampds2_test.csv','ampds2_features.json']:\n (out/name).write_text('ok', encoding='utf-8')\nprint('fake preparation complete')\n""",
        encoding="utf-8",
    )
    (pipeline_dir / "train_forecast.py").write_text(
        """import argparse\np=argparse.ArgumentParser()\nfor arg in ['--pipeline-dir','--out-dir','--gbm','--n-trees','--n-jobs']:\n p.add_argument(arg)\np.add_argument('--no-perm', action='store_true')\np.add_argument('--no-refit', action='store_true')\nargs=p.parse_args()\nprint('fake training complete')\n""",
        encoding="utf-8",
    )
    return PipelineManager(project_root=root, max_workers=1)


def wait_for_job(job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()["job"]
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.025)
    raise AssertionError("Job did not finish within test timeout")


def test_pipeline_catalog_lists_three_tasks():
    response = client.get("/api/v1/pipelines")
    assert response.status_code == 200
    payload = response.json()
    keys = {item["key"] for item in payload["pipelines"]}
    assert keys == {"ampds2", "electricity_theft", "wdseventdb"}
    assert payload["training_is_automatic"] is False


def test_inspect_missing_dataset_is_safe(tmp_path):
    previous = app.state.pipeline_manager
    app.state.pipeline_manager = PipelineManager(project_root=tmp_path)
    try:
        response = client.get("/api/v1/pipelines/ampds2/inspect")
        assert response.status_code == 200
        assert response.json()["files_found"] == 0
    finally:
        app.state.pipeline_manager = previous


def test_prepare_job_calls_pipeline_and_saves_processed_outputs(tmp_path):
    previous = app.state.pipeline_manager
    app.state.pipeline_manager = make_fake_project(tmp_path)
    try:
        response = client.post("/api/v1/pipelines/ampds2/prepare", json={"horizon_hours": 2})
        assert response.status_code == 202, response.text
        job_id = response.json()["job"]["job_id"]
        job = wait_for_job(job_id)
        assert job["status"] == "succeeded", job
        assert "fake preparation complete" in job["stdout"]
        processed = tmp_path / "data" / "processed" / "ampds2"
        assert (processed / "ampds2_train.csv").is_file()
        assert (processed / "ampds2_features.json").is_file()
    finally:
        app.state.pipeline_manager = previous


def test_training_requires_prepared_files(tmp_path):
    previous = app.state.pipeline_manager
    app.state.pipeline_manager = make_fake_project(tmp_path)
    try:
        response = client.post("/api/v1/pipelines/ampds2/train", json={})
        assert response.status_code == 400
        assert "Run the prepare endpoint first" in response.json()["detail"]
    finally:
        app.state.pipeline_manager = previous


def test_explicit_training_job_is_queued_only_after_preparation(tmp_path):
    previous = app.state.pipeline_manager
    app.state.pipeline_manager = make_fake_project(tmp_path)
    try:
        prep = client.post("/api/v1/pipelines/ampds2/prepare", json={})
        assert prep.status_code == 202
        assert wait_for_job(prep.json()["job"]["job_id"])["status"] == "succeeded"
        train = client.post("/api/v1/pipelines/ampds2/train", json={"n_trees": 20})
        assert train.status_code == 202, train.text
        job = wait_for_job(train.json()["job"]["job_id"])
        assert job["status"] == "succeeded", job
        assert "fake training complete" in job["stdout"]
    finally:
        app.state.pipeline_manager = previous
