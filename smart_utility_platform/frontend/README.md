# Streamlit Frontend

A polished control-room interface for the Smart Energy, Water & Utility Intelligence Platform. The frontend communicates with FastAPI over HTTP; it does not load datasets, train models, or run ML inference in the Streamlit process.

## Start

From the project root, install dependencies and start the API and UI in separate terminals:

```bash
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

```bash
streamlit run frontend/app.py
```

The default API base URL is `http://127.0.0.1:8000`. Set `BACKEND_URL` to override it, or change it from the sidebar.

## Workspace pages

- **Overview:** live dataset readiness, pipeline preparation and model artifact status.
- **Data & pipelines:** raw file inventory, sampled schema/missingness inspection, and explicit dataset preparation.
- **Jobs & results:** monitor job states/logs and view saved evaluation output.
- **Model studio:** explicitly submit training only when prepared data is ready; submit inference records only when real model artifacts are installed.
- **API & help:** service endpoints and local run commands.

Metrics are sourced from the backend. No example prediction, model score, or synthetic usage trend is presented as a real result.
