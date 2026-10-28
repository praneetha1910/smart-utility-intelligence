# Backend (FastAPI)

The backend provides APIs for the frontend and safely orchestrates the three ML dataset pipelines.
It does not start data preparation or model training on application startup.

## Run

From the project root:

```bash
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API explorer.

## Pipeline API

- `GET /api/v1/pipelines` — pipeline catalog and readiness.
- `GET /api/v1/datasets` — raw files, processed split readiness, and result availability.
- `GET /api/v1/pipelines/{key}/inspect?sample_rows=200` — sample schema inspection; does not modify sources.
- `POST /api/v1/pipelines/{key}/prepare` — explicitly queue the selected cleaning/split pipeline.
- `GET /api/v1/jobs/{job_id}` — poll a long-running job.
- `GET /api/v1/jobs` — recently submitted jobs in this server process.
- `POST /api/v1/pipelines/{key}/train` — explicitly queue training after the prepared files exist. This is for a later phase and is never automatic.
- `GET /api/v1/pipelines/{key}/results` — saved training metrics, if a training run exists.

Allowed pipeline keys: `ampds2`, `electricity_theft`, `wdseventdb`.

## Source file locations

- `data/raw/ampds2/`: `Electricity_WHE.csv` and `Climate_HourlyWeather.csv` (additional meter files are optional).
- `data/raw/electricity_theft/`: the extracted SGCC dataset CSV.
- `data/raw/wdseventdb/`: extracted WDSEventDB recording CSV/Excel files.

The API does not accept arbitrary commands or external filesystem paths. It maps known pipeline names to scripts under `ml/pipelines/`. Raw datasets are read but not modified. Prepared files go to `data/processed/{pipeline_key}/`; training artifacts and metrics go to `output/models/{pipeline_key}/`.

## Important note about predictions

The existing `/api/v1/forecast`, `/api/v1/theft/predict`, and `/api/v1/water-leakage/predict` endpoints use the inference artifact contract in `ml/inference.py`. The current dataset training scripts save run-specific artifacts under `output/models/{pipeline_key}/models/`; promotion/adapter of a selected trained artifact into the inference contract is a separate step and is not performed automatically.
