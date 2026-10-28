# Smart Energy, Water & Utility Intelligence Platform

A modular application separated into three concerns: **frontend**, **backend**, and **ML**. The backend now integrates the supplied AMPds2, SGCC theft, and WDSEventDB dataset pipelines. Model training is only started by an explicit API request; no training starts at backend startup.

## Project map

```text
smart_utility_platform/
├── frontend/                 # Streamlit UI (kept separate from API/ML logic)
├── backend/                  # FastAPI routes, schemas, service adapters
│   ├── api/routes/           # health, dashboard, predictions, pipeline lifecycle
│   ├── schemas/              # Pydantic request/response models
│   └── services/             # ML gateway and safe pipeline orchestrator
├── ml/
│   ├── data/                 # generic dataset inspection/loaders
│   ├── pipelines/            # supplied dataset cleaning and training scripts
│   ├── inference.py          # inference against promoted model artifacts
│   ├── models/               # inference artifacts when promoted
│   └── tests/                # generic data/inference tests
├── data/raw/                 # source datasets, never modified in place
├── data/processed/           # generated clean splits/features
├── output/models/            # model-comparison run outputs and artifacts
├── configs/                  # dataset configuration
├── reports/                  # dataset audits and job logs
├── requirements.txt
└── pyproject.toml
```

## Responsibilities

- **Frontend:** calls the API and presents information; it does not clean data or train models.
- **Backend:** validates requests, reports readiness, inspects dataset schemas, and explicitly orchestrates the known ML scripts as background jobs.
- **ML:** owns dataset-specific cleaning/splitting, model training/evaluation, and inference.

## Setup

Use Python 3.10–3.12 and run from the project root:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The pipeline layer has optional LightGBM/XGBoost acceleration. It also supports a scikit-learn histogram gradient-boosting fallback. Install `ml/pipelines/requirements.txt` only if you specifically want the optional booster libraries.

## Run the Streamlit frontend

Start the frontend in a second terminal from the project root, after the API is running:

```bash
streamlit run frontend/app.py
```

The default API URL is `http://127.0.0.1:8000`. The sidebar lets you change it if the API is hosted elsewhere. The interface includes live readiness cards, dataset inspection, explicit preparation jobs, job logs, saved evaluation results, explicit training controls, and model inference forms. It uses backend-returned data only; no placeholder consumption metrics or fake predictions are presented as actual results.

## Add the datasets

- `data/raw/ampds2/`: at minimum `Electricity_WHE.csv` and `Climate_HourlyWeather.csv`.
- `data/raw/electricity_theft/`: the extracted SGCC dataset CSV (the repository's split archive parts must be extracted together first).
- `data/raw/wdseventdb/`: extracted WDSEventDB CSV/Excel recording files.

Raw files are not bundled with the project. Keep original dataset files unchanged.

## Run the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

API documentation: `http://127.0.0.1:8000/docs`.

## Backend pipeline endpoints

The API pipeline keys are `ampds2`, `electricity_theft`, and `wdseventdb`.

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/pipelines` | List pipelines and their readiness |
| `GET /api/v1/datasets` | Inventory raw files, prepared splits and training results |
| `GET /api/v1/pipelines/{key}/inspect?sample_rows=200` | Inspect sampled columns and missingness without changing inputs |
| `POST /api/v1/pipelines/{key}/prepare` | Explicitly run that dataset's cleaning/splitting script |
| `GET /api/v1/jobs/{job_id}` | Poll an asynchronous preparation/training job |
| `GET /api/v1/jobs` | List jobs submitted to the current server process |
| `POST /api/v1/pipelines/{key}/train` | Explicitly run training after prepared files exist; not called automatically |
| `GET /api/v1/pipelines/{key}/results` | Read saved model metrics/comparison table when available |

Example: prepare AMPds2 after placing source files:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/pipelines/ampds2/prepare \\
  -H 'Content-Type: application/json' \\
  -d '{"target_meter":"WHE","horizon_hours":1}'
```

The API returns a `job_id`. Poll `GET /api/v1/jobs/{job_id}` until `status` is `succeeded` or `failed`. Preparation outputs go to `data/processed/{key}/`. Training outputs go to `output/models/{key}/`.

The theft prepare endpoint accepts `csv_filename` relative to `data/raw/electricity_theft/` when there are multiple CSV files. WDSEventDB preparation scans the extracted recordings beneath its dataset folder.

## Training is not automatic

This project is currently in the backend-integration phase. The `POST .../train` endpoints are implemented for the later training stage, but they have **not been called** as part of this backend implementation. Training requires successful preparation outputs first.

The three training scripts save per-run model artifacts in `output/models/{key}/models/` and evaluation output in `output/models/{key}/`. The existing prediction endpoints currently consume a separate artifact contract under `ml/models/`; promoting the selected model and defining each module's serving-time feature contract is a follow-up integration step after training and model selection. This prevents an arbitrary candidate model from being served before evaluation.

## Current API prediction endpoints

- `GET /api/v1/models/status`
- `POST /api/v1/forecast`
- `POST /api/v1/theft/predict`
- `POST /api/v1/water-leakage/predict`

These endpoints report model-not-ready until inference artifacts and feature metadata are installed. They do not produce fake predictions.

## Tests

```bash
pytest -q
```

Current tests cover the backend health/inference routes, the data-preprocessing helpers, pipeline registry, sampled inspection behavior, asynchronous job execution using a test script, and the requirement that training needs prepared splits. The real data/model training scripts are not run by the default backend tests.

## Data and evaluation cautions

- AMPds2 represents one residence, not an entire college campus.
- Validate the exact SGCC label and customer ID columns after inspecting the real file.
- WDSEventDB includes normal, leakage, cyberattack, and sensor-failure scenarios; do not describe every event as a leak.
- Forecast validation must be chronological and fit transformations on training data only.
- A model alert is a signal for investigation, not proof of theft or confirmation of a physical leak.
