"""Safe orchestration layer for dataset inspection, preprocessing, and training CLIs.

The API never accepts an arbitrary command or filesystem path. It maps known pipeline
keys to scripts within this repository and only passes validated arguments.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Any
from uuid import uuid4

import pandas as pd

from ml.data.loaders import load_table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINES_DIR = PROJECT_ROOT / "ml" / "pipelines"
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
MODEL_OUTPUT_ROOT = PROJECT_ROOT / "output" / "models"
SUPPORTED_PIPELINE_SUFFIXES = {".csv", ".xlsx", ".xls", ".xlsm"}
MAX_INSPECT_FILES = 50
MAX_LOG_CHARS = 24000


@dataclass(frozen=True)
class PipelineSpec:
    key: str
    name: str
    description: str
    script: str
    training_script: str
    raw_dir_name: str
    processed_dir_name: str
    model_dir_name: str
    required_prepared_files: tuple[str, ...]
    training_results_file: str


PIPELINE_SPECS: dict[str, PipelineSpec] = {
    "ampds2": PipelineSpec(
        key="ampds2", name="AMPds2 electricity forecasting",
        description="Cleans minute-level electricity and weather data and makes chronological splits.",
        script="ampds2_pipeline.py", training_script="train_forecast.py",
        raw_dir_name="ampds2", processed_dir_name="ampds2", model_dir_name="ampds2",
        required_prepared_files=("ampds2_train.csv", "ampds2_val.csv", "ampds2_test.csv", "ampds2_features.json"),
        training_results_file="forecast_results.json",
    ),
    "electricity_theft": PipelineSpec(
        key="electricity_theft", name="SGCC electricity theft detection",
        description="Cleans customer consumption and builds customer-level stratified splits/features.",
        script="theft_pipeline.py", training_script="train_theft.py",
        raw_dir_name="electricity_theft", processed_dir_name="electricity_theft", model_dir_name="electricity_theft",
        required_prepared_files=("theft_features_train.csv", "theft_features_val.csv", "theft_features_test.csv"),
        training_results_file="theft_results.json",
    ),
    "wdseventdb": PipelineSpec(
        key="wdseventdb", name="WDSEventDB water-system event detection",
        description="Cleans sensor recordings, creates causal features, and splits by recording.",
        script="wdseventdb_pipeline.py", training_script="train_wds.py",
        raw_dir_name="wdseventdb", processed_dir_name="wdseventdb", model_dir_name="wdseventdb",
        required_prepared_files=("wds_train.csv", "wds_test.csv", "wds_schema.json"),
        training_results_file="wds_results.json",
    ),
}


class PipelineServiceError(RuntimeError):
    """Expected orchestration/validation error suitable for an HTTP response."""


class PipelineConflictError(PipelineServiceError):
    """A job already operates on this dataset."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable_job(job: dict[str, Any]) -> dict[str, Any]:
    return {**job}


class PipelineManager:
    """Inventory datasets and start inspect/preprocess/train jobs in worker threads."""

    def __init__(self, project_root: str | Path = PROJECT_ROOT, max_workers: int = 2) -> None:
        self.project_root = Path(project_root).resolve()
        self.pipelines_dir = self.project_root / "ml" / "pipelines"
        self.raw_root = self.project_root / "data" / "raw"
        self.processed_root = self.project_root / "data" / "processed"
        self.model_output_root = self.project_root / "output" / "models"
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="utility-pipeline")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active_dataset_jobs: dict[str, str] = {}
        self._lock = threading.RLock()

    def _spec(self, key: str) -> PipelineSpec:
        try:
            return PIPELINE_SPECS[key]
        except KeyError as exc:
            raise PipelineServiceError(
                f"Unknown pipeline '{key}'. Allowed values: {', '.join(PIPELINE_SPECS)}"
            ) from exc

    def _raw_dir(self, spec: PipelineSpec) -> Path:
        return self.raw_root / spec.raw_dir_name

    def _processed_dir(self, spec: PipelineSpec) -> Path:
        return self.processed_root / spec.processed_dir_name

    def _model_dir(self, spec: PipelineSpec) -> Path:
        return self.model_output_root / spec.model_dir_name

    def _dataset_files(self, spec: PipelineSpec) -> list[Path]:
        root = self._raw_dir(spec)
        if not root.is_dir():
            return []
        files = [p for p in root.rglob("*") if p.is_file()
                 and p.suffix.lower() in SUPPORTED_PIPELINE_SUFFIXES
                 and not p.name.startswith((".", "~$"))]
        return sorted(files)

    def _expected_inputs(self, spec: PipelineSpec, target_meter: str = "WHE") -> list[str]:
        if spec.key == "ampds2":
            return [f"Electricity_{target_meter}.csv", "Climate_HourlyWeather.csv"]
        if spec.key == "electricity_theft":
            return ["One SGCC customer-consumption CSV containing ID and label columns"]
        return ["At least one WDSEventDB CSV/Excel recording"]

    def inventory(self) -> dict[str, Any]:
        """Return dataset availability and generated outputs for all three pipelines."""
        datasets: dict[str, Any] = {}
        for key, spec in PIPELINE_SPECS.items():
            raw_dir = self._raw_dir(spec)
            files = self._dataset_files(spec)
            listing = []
            for path in files[:MAX_INSPECT_FILES]:
                stat = path.stat()
                listing.append({
                    "path": path.relative_to(raw_dir).as_posix(),
                    "name": path.name,
                    "suffix": path.suffix.lower(),
                    "size_bytes": stat.st_size,
                })
            prepared_dir = self._processed_dir(spec)
            prepared_missing = [name for name in spec.required_prepared_files
                                if not (prepared_dir / name).is_file()]
            source_available = self._source_available(spec, files, target_meter="WHE")
            datasets[key] = {
                "name": spec.name,
                "raw_directory": str(raw_dir),
                "raw_directory_exists": raw_dir.is_dir(),
                "raw_file_count": len(files),
                "raw_files_truncated": len(files) > MAX_INSPECT_FILES,
                "raw_files": listing,
                "expected_inputs": self._expected_inputs(spec),
                "source_ready": source_available,
                "prepared_directory": str(prepared_dir),
                "prepared_ready": not prepared_missing,
                "missing_prepared_files": prepared_missing,
                "training_output_directory": str(self._model_dir(spec)),
                "training_results_available": (self._model_dir(spec) / spec.training_results_file).is_file(),
                "pipeline_script_available": (self.pipelines_dir / spec.script).is_file(),
                "training_script_available": (self.pipelines_dir / spec.training_script).is_file(),
            }
        return {"project_root": str(self.project_root), "datasets": datasets}

    def _source_available(self, spec: PipelineSpec, files: list[Path] | None = None,
                          target_meter: str = "WHE") -> bool:
        raw_dir = self._raw_dir(spec)
        files = files if files is not None else self._dataset_files(spec)
        if spec.key == "ampds2":
            return all((raw_dir / required).is_file() for required in self._expected_inputs(spec, target_meter))
        if spec.key == "electricity_theft":
            return any(p.suffix.lower() == ".csv" for p in files)
        return any(p.suffix.lower() in SUPPORTED_PIPELINE_SUFFIXES for p in files)

    def list_pipelines(self) -> dict[str, Any]:
        inventory = self.inventory()["datasets"]
        return {
            "pipelines": [
                {
                    "key": key,
                    "name": spec.name,
                    "description": spec.description,
                    "source_ready": inventory[key]["source_ready"],
                    "prepared_ready": inventory[key]["prepared_ready"],
                    "training_results_available": inventory[key]["training_results_available"],
                    "prepare_endpoint": f"POST /api/v1/pipelines/{key}/prepare",
                    "train_endpoint": f"POST /api/v1/pipelines/{key}/train",
                }
                for key, spec in PIPELINE_SPECS.items()
            ],
            "training_is_automatic": False,
            "note": "Preparation and training run only when their respective endpoints are explicitly called.",
        }

    def inspect_dataset(self, key: str, sample_rows: int = 200) -> dict[str, Any]:
        """Inspect local schemas using samples; never modify input files."""
        spec = self._spec(key)
        files = self._dataset_files(spec)
        if not files:
            return {
                "pipeline": key,
                "raw_directory": str(self._raw_dir(spec)),
                "files_found": 0,
                "message": "No supported source files found. Add the dataset files then inspect again.",
                "files": [],
            }

        inspected: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        for path in files[:MAX_INSPECT_FILES]:
            try:
                frame = load_table(path, nrows=sample_rows)
                columns = []
                for column in frame.columns[:150]:
                    series = frame[column]
                    columns.append({
                        "name": str(column),
                        "dtype_sample": str(series.dtype),
                        "null_fraction_in_sample": round(float(series.isna().mean()), 4) if len(series) else None,
                        "unique_values_in_sample": int(series.nunique(dropna=True)),
                    })
                inspected.append({
                    "path": path.relative_to(self._raw_dir(spec)).as_posix(),
                    "name": path.name,
                    "rows_sampled": int(len(frame)),
                    "column_count": int(len(frame.columns)),
                    "columns_truncated": len(frame.columns) > 150,
                    "columns": columns,
                    "sample_only": True,
                })
            except Exception as exc:
                failures.append({"path": path.relative_to(self._raw_dir(spec)).as_posix(), "error": str(exc)})
        return {
            "pipeline": key,
            "sample_rows_requested": sample_rows,
            "files_found": len(files),
            "files_inspected": len(inspected),
            "files_truncated": len(files) > MAX_INSPECT_FILES,
            "files": inspected,
            "failures": failures,
            "source_files_modified": False,
            "note": "Schema and missingness values are sample-based; they are not full-file statistics.",
        }

    def _select_theft_csv(self, spec: PipelineSpec, csv_filename: str | None) -> Path:
        raw_dir = self._raw_dir(spec).resolve()
        if csv_filename:
            selected = (raw_dir / csv_filename).resolve()
            if not selected.is_relative_to(raw_dir):
                raise PipelineServiceError("csv_filename must remain inside the electricity_theft data directory.")
            if not selected.is_file() or selected.suffix.lower() != ".csv":
                raise PipelineServiceError(f"Requested CSV does not exist or is not a CSV: {csv_filename}")
            return selected
        files = [p for p in self._dataset_files(spec) if p.suffix.lower() == ".csv"]
        if len(files) == 1:
            return files[0]
        if not files:
            raise PipelineServiceError("No SGCC CSV found. Put the extracted dataset CSV in data/raw/electricity_theft/.")
        raise PipelineServiceError(
            "More than one CSV was found. Pass csv_filename relative to data/raw/electricity_theft/."
        )

    def _ensure_source_inputs(self, spec: PipelineSpec, params: dict[str, Any]) -> tuple[list[str], Path]:
        raw_dir = self._raw_dir(spec)
        out_dir = self._processed_dir(spec)
        script_path = self.pipelines_dir / spec.script
        if not script_path.is_file():
            raise PipelineServiceError(f"Pipeline script missing: {script_path}")
        if spec.key == "ampds2":
            target_meter = params.get("target_meter", "WHE")
            submeters = params.get("submeters", [])
            if not re.fullmatch(r"[A-Za-z0-9_]{1,20}", target_meter):
                raise PipelineServiceError("target_meter may contain only letters, numbers, and underscores.")
            if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,20}", name) for name in submeters):
                raise PipelineServiceError("Each submeter name may contain only letters, numbers, and underscores.")
            required = self._expected_inputs(spec, target_meter)
            missing = [name for name in required if not (raw_dir / name).is_file()]
            if missing:
                raise PipelineServiceError(f"Missing AMPds2 input file(s) in {raw_dir}: {', '.join(missing)}")
            command = [sys.executable, "-u", str(script_path), "--data-dir", str(raw_dir),
                       "--out-dir", str(out_dir), "--target-meter", target_meter,
                       "--horizon", str(params.get("horizon_hours", 1))]
            submeters = params.get("submeters", [])
            if submeters:
                command.extend(["--submeters", *submeters])
        elif spec.key == "electricity_theft":
            csv_path = self._select_theft_csv(spec, params.get("csv_filename"))
            command = [sys.executable, "-u", str(script_path), "--csv", str(csv_path), "--out-dir", str(out_dir)]
            if params.get("keep_all_zero_customers"):
                command.append("--keep-all-zero")
            if params.get("disable_winsorization"):
                command.append("--no-winsor")
        else:
            files = self._dataset_files(spec)
            if not files:
                raise PipelineServiceError(f"No WDSEventDB recording files found under {raw_dir}")
            command = [sys.executable, "-u", str(script_path), "run", "--data-dir", str(raw_dir), "--out-dir", str(out_dir)]
        return command, out_dir

    def _training_inputs(self, spec: PipelineSpec) -> None:
        missing = [name for name in spec.required_prepared_files if not (self._processed_dir(spec) / name).is_file()]
        if missing:
            raise PipelineServiceError(
                f"Cannot train {spec.key}: prepared data is missing from {self._processed_dir(spec)}: "
                f"{', '.join(missing)}. Run the prepare endpoint first."
            )
        if not (self.pipelines_dir / spec.training_script).is_file():
            raise PipelineServiceError(f"Training script missing: {self.pipelines_dir / spec.training_script}")

    def _build_training_command(self, spec: PipelineSpec, params: dict[str, Any]) -> list[str]:
        self._training_inputs(spec)
        pipeline_dir = self._processed_dir(spec)
        out_dir = self._model_dir(spec)
        script = self.pipelines_dir / spec.training_script
        command = [sys.executable, "-u", str(script), "--pipeline-dir", str(pipeline_dir),
                   "--out-dir", str(out_dir), "--gbm", params.get("gbm", "auto"),
                   "--n-trees", str(params.get("n_trees", 200)), "--n-jobs", str(params.get("n_jobs", 1))]
        if params.get("skip_permutation_importance", True) and spec.key in {"ampds2", "electricity_theft"}:
            command.append("--no-perm")
        if spec.key == "electricity_theft" and params.get("isolation_forest", False):
            command.append("--isolation-forest")
        if spec.key == "wdseventdb":
            command.extend(["--gbm-rounds", str(params.get("gbm_rounds", 100)),
                            "--train-stride", str(params.get("train_stride", 1)),
                            "--cv-max-folds", str(params.get("cv_max_folds", 4))])
        if spec.key == "ampds2" and params.get("no_refit", False):
            command.append("--no-refit")
        return command

    def submit(self, key: str, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        spec = self._spec(key)
        if kind not in {"prepare", "train"}:
            raise PipelineServiceError("Job kind must be 'prepare' or 'train'.")
        if kind == "prepare":
            command, _ = self._ensure_source_inputs(spec, params)
            timeout = 6 * 60 * 60
        else:
            command = self._build_training_command(spec, params)
            timeout = 12 * 60 * 60

        with self._lock:
            active_job_id = self._active_dataset_jobs.get(key)
            if active_job_id:
                active = self._jobs.get(active_job_id, {})
                if active.get("status") in {"queued", "running"}:
                    raise PipelineConflictError(
                        f"A {active.get('kind')} job for {key} is already {active.get('status')} "
                        f"(job_id={active_job_id})."
                    )
            job_id = uuid4().hex
            job = {
                "job_id": job_id,
                "pipeline": key,
                "kind": kind,
                "status": "queued",
                "created_at": _now(),
                "started_at": None,
                "finished_at": None,
                "return_code": None,
                "command": command,
                "stdout": "",
                "stderr": "",
                "error": None,
            }
            self._jobs[job_id] = job
            self._active_dataset_jobs[key] = job_id
            self._executor.submit(self._run_job, job_id, command, timeout)
            return _jsonable_job(job)

    def _run_job(self, job_id: str, command: list[str], timeout: int) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "running"
            job["started_at"] = _now()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            result = subprocess.run(
                command,
                cwd=self.pipelines_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            with self._lock:
                job["return_code"] = result.returncode
                job["stdout"] = result.stdout[-MAX_LOG_CHARS:]
                job["stderr"] = result.stderr[-MAX_LOG_CHARS:]
                job["status"] = "succeeded" if result.returncode == 0 else "failed"
                if result.returncode != 0:
                    job["error"] = f"Pipeline exited with return code {result.returncode}."
                job["finished_at"] = _now()
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes): stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes): stderr = stderr.decode(errors="replace")
            with self._lock:
                job["status"] = "failed"
                job["error"] = f"Pipeline timed out after {timeout} seconds."
                job["stdout"] = str(stdout)[-MAX_LOG_CHARS:]
                job["stderr"] = str(stderr)[-MAX_LOG_CHARS:]
                job["finished_at"] = _now()
        except Exception as exc:
            with self._lock:
                job["status"] = "failed"
                job["error"] = str(exc)
                job["finished_at"] = _now()

        with self._lock:
            if self._active_dataset_jobs.get(job["pipeline"]) == job_id:
                del self._active_dataset_jobs[job["pipeline"]]
            # Persist job detail so a later project run can inspect the log/result.
            logs_dir = self.project_root / "reports" / "pipeline_jobs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / f"{job_id}.json"
            safe_job = {k: v for k, v in job.items() if k != "command"}
            log_path.write_text(json.dumps(safe_job, indent=2), encoding="utf-8")

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return _jsonable_job(job)
        log_path = self.project_root / "reports" / "pipeline_jobs" / f"{job_id}.json"
        if log_path.is_file():
            try:
                return json.loads(log_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def get_training_results(self, key: str) -> dict[str, Any]:
        spec = self._spec(key)
        results_path = self._model_dir(spec) / spec.training_results_file
        comparison_candidates = [
            self._model_dir(spec) / "forecast_comparison.csv",
            self._model_dir(spec) / "theft_comparison.csv",
            self._model_dir(spec) / "wds_comparison.csv",
        ]
        comparison_path = next((p for p in comparison_candidates if p.is_file()), None)
        if not results_path.is_file():
            return {
                "pipeline": key,
                "available": False,
                "message": "No training results yet. Training has not been run or completed successfully.",
                "results_path": str(results_path),
            }
        try:
            result_data = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PipelineServiceError(f"Could not read training results: {exc}") from exc
        comparison = None
        if comparison_path:
            try:
                comparison = pd.read_csv(comparison_path).to_dict(orient="records")
            except Exception:
                comparison = None
        return {
            "pipeline": key,
            "available": True,
            "results": result_data,
            "comparison_table": comparison,
            "results_path": str(results_path),
        }

    def recent_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(
                (_jsonable_job(job) for job in self._jobs.values()),
                key=lambda job: job["created_at"], reverse=True,
            )[:limit]
