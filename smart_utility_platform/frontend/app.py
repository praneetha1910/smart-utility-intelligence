"""Streamlit control room for the Smart Energy, Water & Utility Intelligence Platform.

The UI communicates with FastAPI over HTTP. It never loads or trains ML models in
this process; dataset preparation, training and inference are delegated to the backend.
"""
from __future__ import annotations

import html
import json
import os
from datetime import datetime
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# App configuration and design system
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Utility Intelligence | Control Room",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
PIPELINE_META: dict[str, dict[str, str]] = {
    "ampds2": {
        "title": "Electricity forecasting",
        "short": "AMPds2",
        "description": "Understand household electricity patterns and forecast future demand.",
        "icon": "⚡",
        "color": "#35D0BA",
        "evaluation": "MAE gap vs. baseline",
    },
    "electricity_theft": {
        "title": "Electricity theft detection",
        "short": "SGCC",
        "description": "Identify customer consumption patterns that warrant investigation.",
        "icon": "◈",
        "color": "#F6B94A",
        "evaluation": "PR-AUC gap vs. logistic baseline",
    },
    "wdseventdb": {
        "title": "Water-system event detection",
        "short": "WDSEventDB",
        "description": "Evaluate leakage signals alongside other water-network events.",
        "icon": "💧",
        "color": "#76A9FF",
        "evaluation": "Per-recording evaluation",
    },
}
MODEL_META: dict[str, dict[str, str]] = {
    "electricity_forecasting": {"label": "Electricity forecasting", "pipeline": "ampds2", "endpoint": "/api/v1/forecast", "icon": "⚡"},
    "electricity_theft_detection": {"label": "Electricity theft detection", "pipeline": "electricity_theft", "endpoint": "/api/v1/theft/predict", "icon": "◈"},
    "water_leakage_detection": {"label": "Water event detection", "pipeline": "wdseventdb", "endpoint": "/api/v1/water-leakage/predict", "icon": "💧"},
}

st.markdown(
    """
    <style>
    :root {
      --canvas: #0b1220;
      --surface: #111c2d;
      --surface-2: #162338;
      --line: rgba(157, 178, 204, .16);
      --text: #e8eff8;
      --muted: #93a5bc;
      --teal: #35d0ba;
      --blue: #76a9ff;
    }
    html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .stApp { background: radial-gradient(ellipse at 8% 0%, rgba(31, 91, 112, .20), transparent 34%), var(--canvas); color: var(--text); }
    [data-testid="stHeader"] { background: rgba(11,18,32,.80); }
    [data-testid="stSidebar"] { background: #0d1727; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #a9b9cd; }
    [data-testid="stSidebar"] .stRadio label { color: #d9e4f2; font-size: .92rem; padding-top: .16rem; padding-bottom: .16rem; }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }
    h1, h2, h3 { color: #edf4fd !important; letter-spacing: -.025em; }
    h1 { font-size: 2.05rem !important; font-weight: 740 !important; }
    h2 { font-size: 1.35rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.06rem !important; font-weight: 680 !important; }
    p, li, label, .stMarkdown { color: #d6e1ee; }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stMetric"] { background: transparent; }
    [data-testid="stMetricLabel"] { color: #96a9c0; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #f4f8fd; font-weight: 750; }
    [data-testid="stMetricDelta"] { font-size: .78rem; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line) !important; background: rgba(17,28,45,.66); border-radius: 16px; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] { background: #101c2e !important; border-color: #2a3b51 !important; }
    input, textarea { color: #eaf2fc !important; }
    [data-testid="stTextInput"] label, [data-testid="stTextArea"] label, [data-testid="stSelectbox"] label,
    [data-testid="stNumberInput"] label, [data-testid="stCheckbox"] label { color: #b6c6d9 !important; }
    .stButton button, .stDownloadButton button { border-radius: 9px; min-height: 2.55rem; font-weight: 650; transition: all .15s ease; }
    .stButton button[kind="primary"] { background: linear-gradient(100deg, #2bbda9, #2e9fba); border: 0; color: #041b22; }
    .stButton button[kind="secondary"] { background: #17263a; color: #e7f0fb; border: 1px solid #31445d; }
    .stButton button:hover { border-color: #35d0ba; transform: translateY(-1px); }
    a { color: #70dece !important; }
    hr { border-color: var(--line); }
    [data-testid="stAlert"] { border-radius: 12px; }
    .brand { display:flex; gap:12px; align-items:center; padding: .35rem .25rem 1.15rem; }
    .brand-mark { width:43px; height:43px; border-radius:13px; display:flex; align-items:center; justify-content:center; font-size:22px; color:#061b23; background:linear-gradient(145deg,#45dfc6,#83b7ff); box-shadow:0 8px 26px rgba(53,208,186,.16); }
    .brand-name { color:#f2f7fd; font-size:14px; font-weight:760; line-height:1.25; letter-spacing:-.02em; }
    .brand-sub { font-size:10px; color:#7f96b2; letter-spacing:.14em; text-transform:uppercase; margin-top:4px; }
    .hero { position:relative; overflow:hidden; border:1px solid rgba(93,191,191,.23); border-radius:20px; padding:27px 30px; margin-bottom:18px; background:linear-gradient(112deg,rgba(25,68,83,.88),rgba(17,31,51,.94) 54%,rgba(29,43,69,.94)); }
    .hero:after { content:""; position:absolute; width:260px; height:260px; border-radius:50%; right:-70px; top:-130px; border:1px solid rgba(83,220,199,.18); box-shadow:0 0 0 28px rgba(83,220,199,.025),0 0 0 60px rgba(83,220,199,.02); }
    .eyebrow { color:#65dbc8; text-transform:uppercase; font-weight:750; font-size:10px; letter-spacing:.17em; margin-bottom:8px; }
    .hero-title { font-size:29px; line-height:1.17; font-weight:780; letter-spacing:-.04em; color:#f1f7ff; max-width:760px; }
    .hero-copy { color:#a9bed2; font-size:13px; line-height:1.6; max-width:740px; margin-top:10px; }
    .hero-status { display:inline-flex; gap:8px; align-items:center; font-size:11px; border:1px solid rgba(94,220,197,.24); border-radius:999px; color:#90e9d9; padding:6px 10px; background:rgba(25,87,83,.24); margin-top:16px; }
    .status-dot { width:6px; height:6px; border-radius:50%; background:#43e0bc; display:inline-block; }
    .metric-card { height:116px; border:1px solid var(--line); border-radius:15px; background:linear-gradient(145deg,rgba(20,34,54,.96),rgba(16,27,44,.92)); padding:17px 18px; position:relative; overflow:hidden; }
    .metric-top { display:flex; justify-content:space-between; align-items:center; color:#9aacc2; font-size:11px; font-weight:650; }
    .metric-icon { display:flex; width:27px; height:27px; align-items:center; justify-content:center; border-radius:8px; background:rgba(53,208,186,.10); color:#6ce0cf; font-size:14px; }
    .metric-value { color:#f1f6fc; font-size:26px; font-weight:780; letter-spacing:-.04em; line-height:1.25; margin-top:10px; }
    .metric-foot { color:#7f94ad; font-size:10px; margin-top:3px; }
    .section-overline { color:#70d9c8; text-transform:uppercase; letter-spacing:.14em; font-size:10px; font-weight:750; margin-bottom:6px; }
    .pipeline-card { border:1px solid var(--line); border-radius:15px; background:rgba(17,29,47,.82); padding:17px; height:100%; }
    .pipeline-icon { width:38px;height:38px;border-radius:11px;background:rgba(53,208,186,.1);display:flex;align-items:center;justify-content:center;font-size:19px;margin-bottom:12px; }
    .pipeline-title { color:#eaf3fd; font-weight:730; font-size:14px; margin-bottom:6px; }
    .pipeline-copy { color:#91a6bf; font-size:11px; line-height:1.45; min-height:34px; }
    .pipeline-meta { display:flex; flex-wrap:wrap; gap:6px; margin-top:13px; }
    .pill { display:inline-flex; padding:5px 8px; border-radius:7px; border:1px solid var(--line); font-size:10px; font-weight:650; color:#aebed1; background:rgba(255,255,255,.025); }
    .pill.ok { color:#77e4ce; border-color:rgba(53,208,186,.22); background:rgba(53,208,186,.07); }
    .pill.warn { color:#ffcc77; border-color:rgba(246,185,74,.22); background:rgba(246,185,74,.07); }
    .pill.info { color:#a4c6ff; border-color:rgba(118,169,255,.22); background:rgba(118,169,255,.07); }
    .subtle { color:#8ea3bd; font-size:12px; }
    .empty-state { border:1px dashed #34485f; border-radius:14px; padding:24px; text-align:center; background:rgba(20,34,54,.38); }
    .empty-icon { font-size:26px; margin-bottom:8px; }
    .empty-title { color:#e3edf9; font-weight:700; margin-bottom:4px; }
    .empty-copy { color:#91a5bc; font-size:12px; max-width:530px; margin:auto; line-height:1.55; }
    .small-label { color:#8ea3bc; text-transform:uppercase; font-size:10px; letter-spacing:.12em; font-weight:740; }
    @media (max-width: 900px) { .hero-title { font-size:23px; } .block-container { padding-left: 1rem; padding-right: 1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# API client helpers
# -----------------------------------------------------------------------------
def api_request(
    api_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> tuple[Any | None, str | None, int | None]:
    """Call FastAPI and return parsed JSON plus readable error details."""
    url = f"{api_url.rstrip('/')}{path}"
    try:
        response = httpx.request(method, url, json=payload, params=params, timeout=timeout)
        try:
            body = response.json()
        except ValueError:
            body = {"detail": response.text or "The API returned an empty response."}
        if response.is_error:
            detail = body.get("detail", body) if isinstance(body, dict) else body
            return None, f"HTTP {response.status_code}: {detail}", response.status_code
        return body, None, response.status_code
    except httpx.TimeoutException:
        return None, f"The request to {url} timed out. A dataset inspection or job may need more time.", None
    except httpx.HTTPError as exc:
        return None, f"Cannot reach the backend at {api_url}. {exc}", None


def api_get(api_url: str, path: str, params: dict[str, Any] | None = None, timeout: float = 10.0):
    return api_request(api_url, "GET", path, params=params, timeout=timeout)


def api_post(api_url: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0):
    return api_request(api_url, "POST", path, payload=payload or {}, timeout=timeout)


def safe_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def fmt_bytes(size: Any) -> str:
    try:
        value = float(size)
    except (TypeError, ValueError):
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def metric_card(label: str, value: Any, foot: str, icon: str, accent: str = "teal") -> None:
    accent_map = {"teal": "rgba(53,208,186,.10)", "blue": "rgba(118,169,255,.11)", "amber": "rgba(246,185,74,.11)", "violet": "rgba(169,139,255,.11)"}
    icon_color = {"teal": "#6ce0cf", "blue": "#a4c6ff", "amber": "#ffcc77", "violet": "#c4b2ff"}
    bg = accent_map.get(accent, accent_map["teal"])
    fg = icon_color.get(accent, icon_color["teal"])
    st.markdown(
        f'''<div class="metric-card"><div class="metric-top"><span>{html.escape(label)}</span>
        <span class="metric-icon" style="background:{bg};color:{fg}">{html.escape(icon)}</span></div>
        <div class="metric-value">{html.escape(str(value))}</div><div class="metric-foot">{html.escape(foot)}</div></div>''',
        unsafe_allow_html=True,
    )


def section_heading(title: str, subtitle: str | None = None, eyebrow: str | None = None) -> None:
    if eyebrow:
        st.markdown(f'<div class="section-overline">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def empty_state(title: str, copy: str, icon: str = "⌁") -> None:
    st.markdown(
        f'''<div class="empty-state"><div class="empty-icon">{html.escape(icon)}</div>
        <div class="empty-title">{html.escape(title)}</div><div class="empty-copy">{html.escape(copy)}</div></div>''',
        unsafe_allow_html=True,
    )


def status_pill(label: str, mode: str = "info") -> str:
    return f'<span class="pill {mode}">{html.escape(label)}</span>'


def render_pipeline_card(key: str, info: dict[str, Any]) -> None:
    meta = PIPELINE_META.get(key, {"short": key, "title": key, "description": "Dataset pipeline", "icon": "◈"})
    raw_n = safe_count(info.get("raw_file_count"))
    source = bool(info.get("source_ready"))
    prepared = bool(info.get("prepared_ready"))
    results = bool(info.get("training_results_available"))
    badges = (
        status_pill("Source ready" if source else "Source needed", "ok" if source else "warn")
        + status_pill("Prepared" if prepared else "Not prepared", "ok" if prepared else "info")
        + status_pill("Results saved" if results else "Not trained", "ok" if results else "info")
    )
    st.markdown(
        f'''<div class="pipeline-card"><div class="pipeline-icon">{html.escape(meta.get('icon','◈'))}</div>
        <div class="pipeline-title">{html.escape(meta.get('title', info.get('name', key)))}</div>
        <div class="pipeline-copy">{html.escape(meta.get('description',''))}</div>
        <div class="pipeline-meta">{badges}</div>
        <div style="margin-top:14px;color:#8ea3bc;font-size:11px">{raw_n} supported source file(s)</div></div>''',
        unsafe_allow_html=True,
    )


def show_api_error(error: str | None) -> None:
    if error:
        st.error(error)


def pipeline_options(pipelines_data: dict[str, Any] | None, datasets_data: dict[str, Any] | None) -> list[tuple[str, str]]:
    datasets = (datasets_data or {}).get("datasets", {})
    if datasets:
        return [(key, PIPELINE_META.get(key, {}).get("title", info.get("name", key))) for key, info in datasets.items()]
    items = (pipelines_data or {}).get("pipelines", [])
    if items:
        return [(item["key"], item.get("name", item["key"])) for item in items if item.get("key")]
    return [(key, value["title"]) for key, value in PIPELINE_META.items()]


# -----------------------------------------------------------------------------
# Sidebar / navigation
# -----------------------------------------------------------------------------
def navigate_to(page_name: str) -> None:
    """Switch the sidebar navigation through a widget callback."""
    st.session_state["navigation_page"] = page_name


with st.sidebar:
    st.markdown(
        '''<div class="brand"><div class="brand-mark">ϟ</div><div><div class="brand-name">UTILITY<br>INTELLIGENCE</div>
        <div class="brand-sub">Operations control room</div></div></div>''',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="small-label" style="margin:4px 0 8px">Workspace</div>', unsafe_allow_html=True)
    page = st.radio(
        "Workspace navigation",
        ["Overview", "Data & pipelines", "Jobs & results", "Model studio", "API & help"],
        format_func=lambda item: {
            "Overview": "◫   Overview",
            "Data & pipelines": "▤   Data & pipelines",
            "Jobs & results": "◷   Jobs & results",
            "Model studio": "⌘   Model studio",
            "API & help": "⚙   API & help",
        }[item],
        label_visibility="collapsed",
        key="navigation_page",
    )
    st.divider()
    st.markdown('<div class="small-label" style="margin-bottom:8px">Backend connection</div>', unsafe_allow_html=True)
    api_url = st.text_input("Backend URL", value=st.session_state.get("api_url", DEFAULT_API_URL), key="api_url_input").strip().rstrip("/")
    if not api_url:
        api_url = DEFAULT_API_URL
    st.session_state["api_url"] = api_url
    health, health_error, _ = api_get(api_url, "/health", timeout=2.5)
    if health_error:
        st.markdown(status_pill("API offline", "warn"), unsafe_allow_html=True)
        st.caption("Start FastAPI and reconnect.")
    else:
        st.markdown(status_pill("API connected", "ok"), unsafe_allow_html=True)
        st.caption(f"{api_url}")
    if st.button("↻  Refresh workspace", use_container_width=True):
        st.rerun()
    st.divider()
    st.caption("Prototype · Human review required for anomaly alerts")

# Load app state once per rerun. This is live backend data, not synthetic dashboard metrics.
api_url = st.session_state.get("api_url", DEFAULT_API_URL)
dashboard, dashboard_error, _ = api_get(api_url, "/api/v1/dashboard", timeout=8.0)
pipelines_payload, pipelines_error, _ = api_get(api_url, "/api/v1/pipelines")
inventory_payload, inventory_error, _ = api_get(api_url, "/api/v1/datasets")
models_payload, models_error, _ = api_get(api_url, "/api/v1/models/status")
jobs_payload, jobs_error, _ = api_get(api_url, "/api/v1/jobs", params={"limit": 30})

if not isinstance(dashboard, dict):
    dashboard = {}
if not isinstance(pipelines_payload, dict):
    pipelines_payload = {"pipelines": []}
if not isinstance(inventory_payload, dict):
    inventory_payload = {"datasets": {}}
if not isinstance(models_payload, dict):
    models_payload = {"models": {}}
if not isinstance(jobs_payload, dict):
    jobs_payload = {"jobs": []}

pipeline_datasets: dict[str, Any] = dashboard.get("pipeline_datasets") or inventory_payload.get("datasets") or {}
model_statuses: dict[str, Any] = models_payload.get("models") or dashboard.get("models") or {}
all_jobs: list[dict[str, Any]] = jobs_payload.get("jobs", []) or []

# Header shared by all pages.
st.markdown(
    f'''<div class="hero"><div class="eyebrow">SMART UTILITIES · DATA-DRIVEN OPERATIONS</div>
    <div class="hero-title">See the signals.<br>Act before resources are lost.</div>
    <div class="hero-copy">One control room for electricity demand forecasting, suspicious consumption analysis,
    and water-network event diagnosis. Explore datasets, orchestrate pipelines, and inspect model evidence — without
    treating an alert as a verdict.</div>
    <div class="hero-status"><span class="status-dot" style="{'background:#43e0bc' if not dashboard_error else 'background:#ffbd62'}"></span>
    {('Backend connected' if not dashboard_error else 'Backend unavailable')} &nbsp;·&nbsp; {datetime.now().strftime('%d %b %Y')}</div></div>''',
    unsafe_allow_html=True,
)
if dashboard_error:
    st.warning(f"The frontend cannot load live platform status. {dashboard_error}")

# -----------------------------------------------------------------------------
# Page: Overview
# -----------------------------------------------------------------------------
if page == "Overview":
    st.markdown('<div class="section-overline">System overview</div>', unsafe_allow_html=True)
    section_heading("Your utility intelligence at a glance", "These figures are read from the backend inventory and saved model artifacts.")

    dataset_count = len(pipeline_datasets)
    source_ready_count = sum(1 for item in pipeline_datasets.values() if item.get("source_ready"))
    raw_count = sum(safe_count(item.get("raw_file_count")) for item in pipeline_datasets.values())
    prepared_count = sum(1 for item in pipeline_datasets.values() if item.get("prepared_ready"))
    ready_models_count = sum(1 for item in model_statuses.values() if item.get("ready"))
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card("DATA SOURCES READY", f"{source_ready_count}/{dataset_count or 3}", "Required source inputs detected", "▤", "teal")
    with c2:
        metric_card("RAW FILES DISCOVERED", raw_count, "Supported CSV / Excel files", "▧", "blue")
    with c3:
        metric_card("PIPELINES PREPARED", f"{prepared_count}/{dataset_count or 3}", "Required split files available", "↗", "amber")
    with c4:
        metric_card("MODELS READY", f"{ready_models_count}/{len(model_statuses) or 3}", "Fitted artifact + metadata", "⌘", "violet")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        section_heading("Pipeline readiness", "A live view of the three independent analytical tracks.", "OPERATIONS")
        if pipeline_datasets:
            for key, item in pipeline_datasets.items():
                meta = PIPELINE_META.get(key, {})
                raw = safe_count(item.get("raw_file_count"))
                source_ready = bool(item.get("source_ready"))
                prepared = bool(item.get("prepared_ready"))
                trained = bool(item.get("training_results_available"))
                status_text = "Results available" if trained else "Prepared data ready" if prepared else "Waiting for data" if not source_ready else "Ready to prepare"
                status_class = "ok" if trained else "ok" if prepared else "warn" if not source_ready else "info"
                with st.container(border=True):
                    row1, row2 = st.columns([3, 1.2], vertical_alignment="center")
                    with row1:
                        st.markdown(f"**{meta.get('icon','◈')}  {meta.get('title',item.get('name',key))}**")
                        st.caption(f"{raw} raw file(s) · {len(item.get('missing_prepared_files', []))} required prepared output(s) missing")
                    with row2:
                        st.markdown(status_pill(status_text, status_class), unsafe_allow_html=True)
                    progress = (0.25 if source_ready else 0) + (0.35 if prepared else 0) + (0.40 if trained else 0)
                    st.progress(min(progress, 1.0), text="Lifecycle progress — data → preparation → evaluation")
        else:
            empty_state("No pipeline inventory yet", "Start the backend to load dataset and pipeline readiness.", "▤")
    with right:
        section_heading("Model readiness", "Predictions are only enabled when real artifacts are installed.", "INFERENCE")
        for model_key, meta in MODEL_META.items():
            model = model_statuses.get(model_key, {"ready": False, "message": "Model status has not been returned by the API."})
            with st.container(border=True):
                st.markdown(f"**{meta['icon']}  {meta['label']}**")
                st.markdown(status_pill("Ready for inference" if model.get("ready") else "Model not trained", "ok" if model.get("ready") else "warn"), unsafe_allow_html=True)
                st.caption(model.get("version") or model.get("message", "No model version recorded."))
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        st.markdown("**Operational guardrail**")
        st.caption("Forecasts are estimates. Abnormal electricity patterns and water events need human review before any operational decision.")

    section_heading("Recent pipeline jobs", "Latest jobs from the current backend process.", "ACTIVITY")
    if all_jobs:
        job_rows = []
        for job in all_jobs[:8]:
            job_rows.append({
                "Job": job.get("job_id", "")[:10],
                "Pipeline": PIPELINE_META.get(job.get("pipeline"), {}).get("short", job.get("pipeline", "—")),
                "Task": str(job.get("kind", "—")).title(),
                "Status": str(job.get("status", "unknown")).title(),
                "Created": str(job.get("created_at", "—")).replace("T", " ")[:19],
            })
        st.dataframe(pd.DataFrame(job_rows), hide_index=True, use_container_width=True)
    else:
        empty_state("No jobs yet", "After your datasets are in place, inspect them and start preparation from Data & pipelines. No background job is started automatically.", "◷")
    row_a, row_b = st.columns([1, 1])
    with row_a:
        if st.button("Open data & pipelines →", type="primary", use_container_width=True, on_click=navigate_to, args=("Data & pipelines",)):
            pass
    with row_b:
        if st.button("Review model status →", use_container_width=True, on_click=navigate_to, args=("Model studio",)):
            pass

# -----------------------------------------------------------------------------
# Page: Data and pipeline orchestration
# -----------------------------------------------------------------------------
elif page == "Data & pipelines":
    section_heading("Datasets & pipeline control", "Inspect source files first, then prepare deterministic train / validation / test splits.", "DATA WORKSPACE")
    if inventory_error:
        show_api_error(inventory_error)
    choices = pipeline_options(pipelines_payload, inventory_payload)
    labels = {key: f"{PIPELINE_META.get(key, {}).get('icon', '◈')}  {name}" for key, name in choices}
    keys = [key for key, _ in choices]
    selected_key = st.selectbox(
        "Choose a dataset pipeline",
        keys,
        format_func=lambda key: labels.get(key, key),
        key="selected_pipeline_for_data",
    )
    info = pipeline_datasets.get(selected_key, {})
    pmeta = PIPELINE_META.get(selected_key, {})
    pipeline_item = next((x for x in pipelines_payload.get("pipelines", []) if x.get("key") == selected_key), {})
    st.markdown(f"### {pmeta.get('title', info.get('name', selected_key))}")
    st.caption(pmeta.get("description", pipeline_item.get("description", "Inspect and prepare this dataset.")))

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        metric_card("SOURCE FILES", safe_count(info.get("raw_file_count")), "Supported files discovered", "▤", "blue")
    with m2:
        metric_card("SOURCE INPUTS", "Ready" if info.get("source_ready") else "Missing", "Pipeline-specific input check", "✓" if info.get("source_ready") else "!", "teal" if info.get("source_ready") else "amber")
    with m3:
        metric_card("PREPARED DATA", "Ready" if info.get("prepared_ready") else "Pending", "Required outputs present", "↗", "teal" if info.get("prepared_ready") else "blue")
    with m4:
        metric_card("TRAINING RESULTS", "Available" if info.get("training_results_available") else "Not yet", "Saved evaluation artifact", "⌘", "teal" if info.get("training_results_available") else "violet")

    with st.container(border=True):
        st.markdown("**Expected inputs**")
        for expected in info.get("expected_inputs", []):
            st.markdown(f"- `{expected}`")
        st.markdown("**Raw data directory**")
        st.code(info.get("raw_directory", "data/raw/" + selected_key), language=None)
        if not info.get("source_ready"):
            st.warning("The backend has not found all required source inputs. Add the raw dataset files to the directory above and refresh inventory.")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    files = info.get("raw_files", []) or []
    with st.container(border=True):
        section_heading("Source file inventory", "The backend scans supported files and does not modify original inputs.")
        if files:
            table = pd.DataFrame([{
                "File": f.get("name", ""),
                "Relative path": f.get("path", ""),
                "Format": str(f.get("suffix", "")).upper().replace(".", ""),
                "Size": fmt_bytes(f.get("size_bytes")),
            } for f in files])
            st.dataframe(table, hide_index=True, use_container_width=True)
            if info.get("raw_files_truncated"):
                st.caption("Inventory table is truncated. The backend has more supported files than shown here.")
        else:
            empty_state("No source files found", "Place the downloaded dataset in this pipeline's raw directory. This app does not fabricate or upload dataset contents.", "▧")

    inspect_tab, prepare_tab = st.tabs(["Inspect schema", "Prepare dataset"])
    with inspect_tab:
        colx, coly = st.columns([1, 3], vertical_alignment="bottom")
        with colx:
            sample_rows = st.number_input("Sample rows per file", min_value=10, max_value=2000, value=200, step=50)
        with coly:
            inspect_clicked = st.button("Inspect columns & missingness", type="primary", use_container_width=True, key="inspect_dataset_btn")
        if inspect_clicked:
            with st.spinner("Reading samples and profiling columns…"):
                data, error, _ = api_get(api_url, f"/api/v1/pipelines/{selected_key}/inspect", params={"sample_rows": int(sample_rows)}, timeout=45.0)
            if error:
                show_api_error(error)
            else:
                st.session_state[f"inspection::{selected_key}"] = data
        inspection = st.session_state.get(f"inspection::{selected_key}")
        if inspection:
            if inspection.get("files_found", 0) == 0:
                st.info(inspection.get("message", "No supported files found."))
            else:
                st.caption(inspection.get("note", "Inspection statistics are based on sample rows."))
                st.write(f"**Inspected:** {inspection.get('files_inspected', 0)} / {inspection.get('files_found', 0)} file(s)")
                for file in inspection.get("files", []):
                    with st.expander(f"{file.get('name')}  ·  {file.get('rows_sampled')} sampled rows  ·  {file.get('column_count')} columns"):
                        col_records = file.get("columns", [])
                        if col_records:
                            profile = pd.DataFrame(col_records)
                            profile = profile.rename(columns={
                                "name": "Column", "dtype_sample": "Sample dtype",
                                "null_fraction_in_sample": "Missing fraction", "unique_values_in_sample": "Unique values",
                            })
                            if "Missing fraction" in profile.columns:
                                profile["Missing %"] = (pd.to_numeric(profile["Missing fraction"], errors="coerce") * 100).round(2)
                            st.dataframe(profile.drop(columns=["Missing fraction"], errors="ignore"), hide_index=True, use_container_width=True)
                            if "Missing %" in profile.columns:
                                top_missing = profile.sort_values("Missing %", ascending=False).head(12)
                                fig = px.bar(top_missing, x="Missing %", y="Column", orientation="h", title="Sample missingness by column")
                                fig.update_layout(height=310, margin=dict(l=5, r=5, t=45, b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d6e7")
                                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                if inspection.get("failures"):
                    with st.expander(f"Inspection errors ({len(inspection['failures'])})"):
                        st.dataframe(pd.DataFrame(inspection["failures"]), hide_index=True, use_container_width=True)
                st.caption("Schema profiling uses samples, not full-file statistics. Review the actual schema before model training.")
        else:
            empty_state("Inspect before transforming", "The inspection reveals column names, inferred dtypes, sample missingness and unique-value counts. It does not modify files.", "⌕")

    with prepare_tab:
        st.markdown("**Preparation settings**")
        prepare_payload: dict[str, Any] = {}
        if selected_key == "ampds2":
            a, b = st.columns(2)
            with a:
                prepare_payload["target_meter"] = st.text_input("Target meter", value="WHE", help="For example WHE. Must match Electricity_<meter>.csv in the raw directory.", key="prep_target_meter")
            with b:
                prepare_payload["horizon_hours"] = st.number_input("Forecast horizon (hours)", min_value=1, max_value=168, value=1, key="prep_horizon")
            st.caption("The standard input contract expects Electricity_<target meter>.csv and Climate_HourlyWeather.csv in the AMPds2 raw directory.")
        elif selected_key == "electricity_theft":
            csv_paths = [f.get("path") for f in files if str(f.get("suffix", "")).lower() == ".csv"]
            prepare_payload["csv_filename"] = st.selectbox("Customer-consumption CSV", options=[""] + csv_paths, format_func=lambda x: x or "Auto-select (requires exactly one CSV)")
            prepare_payload["keep_all_zero_customers"] = st.checkbox("Keep all-zero customers", value=False)
            prepare_payload["disable_winsorization"] = st.checkbox("Disable winsorization", value=False, help="Changes the pipeline's outlier-handling behavior. Keep default unless you know why it is needed.")
            if not prepare_payload["csv_filename"]:
                prepare_payload["csv_filename"] = None
        else:
            st.info("WDSEventDB preparation discovers supported recording files beneath the raw directory. Keep the event folders intact.")
        st.warning("Preparation writes derived files under data/processed/. Original source files are kept unchanged by design.")
        if st.button("Run dataset preparation", type="primary", use_container_width=True, key="prepare_pipeline_btn", disabled=not bool(info.get("source_ready"))):
            with st.spinner("Submitting preprocessing job to FastAPI…"):
                response, error, _ = api_post(api_url, f"/api/v1/pipelines/{selected_key}/prepare", prepare_payload, timeout=45.0)
            if error:
                show_api_error(error)
            else:
                job = response.get("job", response)
                st.session_state["latest_job_id"] = job.get("job_id")
                st.session_state["latest_job_pipeline"] = selected_key
                st.success(f"Preparation submitted. Job ID: {job.get('job_id', 'unknown')}")
                st.info("The job runs on the backend. Use Jobs & results to monitor status and inspect logs.")
                st.rerun()
        if not info.get("source_ready"):
            st.caption("Add the required source files before preparation can be started.")

# -----------------------------------------------------------------------------
# Page: Jobs and results
# -----------------------------------------------------------------------------
elif page == "Jobs & results":
    section_heading("Jobs & evaluation results", "Monitor explicitly submitted preparation/training work and inspect saved outputs.", "PIPELINE ACTIVITY")
    control1, control2 = st.columns([1, 4], vertical_alignment="center")
    with control1:
        if st.button("↻ Refresh jobs", use_container_width=True):
            st.rerun()
    with control2:
        st.caption("Job status is provided by FastAPI. Long-running tasks continue on the backend while this page is open or closed.")

    if jobs_error:
        show_api_error(jobs_error)
    if all_jobs:
        job_rows = []
        for job in all_jobs:
            job_rows.append({
                "Job ID": job.get("job_id", ""),
                "Pipeline": PIPELINE_META.get(job.get("pipeline"), {}).get("short", job.get("pipeline", "—")),
                "Task": str(job.get("kind", "—")).title(),
                "Status": str(job.get("status", "unknown")).title(),
                "Started": str(job.get("started_at") or job.get("created_at") or "—").replace("T", " ")[:19],
                "Finished": str(job.get("finished_at") or "—").replace("T", " ")[:19],
            })
        jobs_df = pd.DataFrame(job_rows)
        st.dataframe(jobs_df.drop(columns=["Job ID"]), hide_index=True, use_container_width=True)
        job_ids = [job.get("job_id") for job in all_jobs if job.get("job_id")]
        default_job = st.session_state.get("latest_job_id")
        default_index = job_ids.index(default_job) if default_job in job_ids else 0
        selected_job_id = st.selectbox("Select a job to inspect", job_ids, index=default_index, format_func=lambda value: f"{value[:12]}… · {next((j.get('pipeline','') for j in all_jobs if j.get('job_id') == value), '')}")
        a, b = st.columns([1, 4], vertical_alignment="center")
        with a:
            inspect_job_clicked = st.button("Load job detail", type="primary", use_container_width=True)
        if inspect_job_clicked or st.session_state.get("display_job_id") == selected_job_id:
            if inspect_job_clicked:
                st.session_state["display_job_id"] = selected_job_id
            job_detail, job_error, _ = api_get(api_url, f"/api/v1/jobs/{selected_job_id}", timeout=10)
            if job_error:
                show_api_error(job_error)
            elif job_detail:
                job = job_detail.get("job", job_detail)
                status_value = str(job.get("status", "unknown")).lower()
                pill_class = "ok" if status_value == "succeeded" else "warn" if status_value == "failed" else "info"
                st.markdown(status_pill(status_value.upper(), pill_class), unsafe_allow_html=True)
                if job.get("error"):
                    st.error(job["error"])
                detail_a, detail_b, detail_c = st.columns(3)
                detail_a.metric("Pipeline", PIPELINE_META.get(job.get("pipeline"), {}).get("short", job.get("pipeline", "—")))
                detail_b.metric("Task", str(job.get("kind", "—")).title())
                detail_c.metric("Exit code", job.get("return_code") if job.get("return_code") is not None else "—")
                with st.expander("Standard output", expanded=status_value in {"failed", "running"}):
                    st.code(job.get("stdout") or "No standard output captured yet.", language="text")
                with st.expander("Standard error"):
                    st.code(job.get("stderr") or "No standard error captured.", language="text")
                if status_value in {"queued", "running"}:
                    st.info("This job is still active. Refresh this page to load its latest status.")
    else:
        empty_state("No jobs have been submitted", "Use Data & pipelines to inspect your files and submit a preparation job. Training remains a deliberate, separate action.", "◷")

    st.divider()
    section_heading("Saved model evaluation", "Results are shown only when a training run has written its artifacts.", "MODEL EVIDENCE")
    result_choices = pipeline_options(pipelines_payload, inventory_payload)
    result_keys = [key for key, _ in result_choices]
    result_key = st.selectbox("Pipeline results", result_keys, format_func=lambda key: f"{PIPELINE_META.get(key, {}).get('icon','◈')}  {PIPELINE_META.get(key, {}).get('title', key)}", key="results_pipeline_key")
    if st.button("Load saved results", type="primary", key="load_results_btn"):
        result, error, _ = api_get(api_url, f"/api/v1/pipelines/{result_key}/results", timeout=15)
        if error:
            show_api_error(error)
        else:
            st.session_state[f"results::{result_key}"] = result
    result = st.session_state.get(f"results::{result_key}")
    if result:
        if not result.get("available"):
            st.info(result.get("message", "No training results available."))
            if result.get("results_path"):
                st.caption(f"Expected results path: {result['results_path']}")
        else:
            comparison = result.get("comparison_table")
            if comparison:
                comparison_df = pd.DataFrame(comparison)
                st.markdown("**Candidate comparison table**")
                st.dataframe(comparison_df, use_container_width=True)
                numeric_columns = list(comparison_df.select_dtypes(include="number").columns)
                if numeric_columns and len(comparison_df) > 1:
                    metric_name = st.selectbox("Compare metric", numeric_columns, key=f"metric_{result_key}")
                    label_candidates = [c for c in comparison_df.columns if c not in numeric_columns]
                    if label_candidates:
                        label_col = label_candidates[0]
                    else:
                        comparison_df = comparison_df.reset_index(names="Candidate")
                        label_col = "Candidate"
                    fig = px.bar(comparison_df, x=label_col, y=metric_name, title=f"Candidate comparison · {metric_name}")
                    fig.update_layout(height=350, margin=dict(l=8, r=8, t=48, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#c9d6e7")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("**Raw results metadata**")
            st.json(result.get("results", {}), expanded=False)
            st.caption(f"Results source: {result.get('results_path', 'backend results store')}")
    else:
        empty_state("No results loaded", "Load a pipeline's results to see model comparisons and metrics. Results only appear after an actual successful training run.", "▥")

# -----------------------------------------------------------------------------
# Page: Model studio: explicit training and inference
# -----------------------------------------------------------------------------
elif page == "Model studio":
    section_heading("Model studio", "Train only after data preparation is ready; send inference requests only to installed model artifacts.", "MACHINE LEARNING")
    model_tabs = st.tabs(["Training runs", "Prediction studio", "Model inventory"])
    with model_tabs[0]:
        options = pipeline_options(pipelines_payload, inventory_payload)
        pipe_keys = [key for key, _ in options]
        train_key = st.selectbox("Pipeline to train", pipe_keys, format_func=lambda key: f"{PIPELINE_META.get(key, {}).get('icon','◈')}  {PIPELINE_META.get(key, {}).get('title', key)}", key="train_pipeline_key")
        train_info = pipeline_datasets.get(train_key, {})
        ti1, ti2, ti3 = st.columns(3)
        with ti1:
            metric_card("SOURCE INPUTS", "Ready" if train_info.get("source_ready") else "Missing", "Raw data availability", "▤", "teal" if train_info.get("source_ready") else "amber")
        with ti2:
            metric_card("PREPARED DATA", "Ready" if train_info.get("prepared_ready") else "Not ready", "Training requires split files", "↗", "teal" if train_info.get("prepared_ready") else "blue")
        with ti3:
            metric_card("PREVIOUS RESULTS", "Present" if train_info.get("training_results_available") else "None", "Saved run outputs", "▥", "teal" if train_info.get("training_results_available") else "violet")
        if not train_info.get("prepared_ready"):
            st.warning("Training is unavailable until the preprocessing pipeline has produced all required split files.")
            st.caption("Go to Data & pipelines → Prepare dataset, then return here after the job succeeds.")

        with st.form(f"training_form_{train_key}"):
            st.markdown("**Run configuration**")
            left, right = st.columns(2)
            with left:
                gbm = st.selectbox("Gradient-boosting implementation", ["auto", "lightgbm", "xgboost", "hist"], index=0, help="Use auto unless you specifically need to pin a supported implementation.")
                n_trees = st.number_input("Tree count", min_value=10, max_value=2000, value=200, step=50)
                n_jobs = st.number_input("Parallel workers", min_value=1, max_value=8, value=1)
            with right:
                skip_perm = st.checkbox("Skip permutation importance (faster)", value=True)
                isolation_forest = st.checkbox("Include Isolation Forest baseline", value=False, disabled=train_key != "electricity_theft")
                no_refit = st.checkbox("Skip refit on train + validation", value=False, disabled=train_key != "ampds2")
                if train_key == "wdseventdb":
                    gbm_rounds = st.number_input("Boosting rounds", min_value=10, max_value=2000, value=100, step=25)
                    train_stride = st.number_input("Training stride", min_value=1, max_value=1000, value=1)
                    cv_max_folds = st.number_input("Maximum CV folds", min_value=2, max_value=20, value=4)
                else:
                    gbm_rounds, train_stride, cv_max_folds = 100, 1, 4
            st.divider()
            acknowledge = st.checkbox("I confirm this will start a real training job using the prepared local dataset.")
            submitted = st.form_submit_button("Start model training", type="primary", use_container_width=True, disabled=not bool(train_info.get("prepared_ready")))
        if submitted:
            if not acknowledge:
                st.error("Confirm the training action before submitting the job.")
            else:
                payload = {
                    "gbm": gbm, "n_trees": int(n_trees), "n_jobs": int(n_jobs),
                    "skip_permutation_importance": bool(skip_perm), "isolation_forest": bool(isolation_forest),
                    "no_refit": bool(no_refit), "gbm_rounds": int(gbm_rounds),
                    "train_stride": int(train_stride), "cv_max_folds": int(cv_max_folds),
                }
                with st.spinner("Submitting explicit training request to FastAPI…"):
                    response, error, _ = api_post(api_url, f"/api/v1/pipelines/{train_key}/train", payload, timeout=45.0)
                if error:
                    show_api_error(error)
                else:
                    job = response.get("job", response)
                    st.session_state["latest_job_id"] = job.get("job_id")
                    st.session_state["latest_job_pipeline"] = train_key
                    st.success(f"Training job submitted: {job.get('job_id', 'unknown')}")
                    st.info("Monitor it in Jobs & results. No training is started by opening the dashboard.")
        with st.expander("Training safety and evaluation notes"):
            st.markdown("""
            - Forecasting should retain chronological train/validation/test separation.
            - SGCC evaluation should keep customer groups separate and account for class imbalance.
            - WDSEventDB should be evaluated by recording and report which event categories fail.
            - Select models from held-out evaluation evidence, not training score alone.
            - A theft-related flag is not proof of theft; a water-network event is not automatically a confirmed physical leak.
            """)

    with model_tabs[1]:
        statuses = model_statuses
        select_model = st.selectbox("Select a deployed model", list(MODEL_META.keys()), format_func=lambda key: f"{MODEL_META[key]['icon']}  {MODEL_META[key]['label']}", key="inference_model_key")
        meta = MODEL_META[select_model]
        model = statuses.get(select_model, {"ready": False, "message": "Model status is not available."})
        if model.get("ready"):
            st.success(f"{meta['label']} is ready for inference.")
            if model.get("metrics"):
                st.markdown("**Recorded evaluation metadata**")
                st.json(model.get("metrics"), expanded=False)
            feature_cols = model.get("feature_columns", [])
            st.caption("Inference uses the exact feature contract saved with the model. It does not automatically engineer features from raw meter records.")
            st.code(", ".join(feature_cols) if feature_cols else "Feature metadata unavailable", language=None)
            record_json = st.text_area("Feature records as JSON array", value="[]", height=175, key=f"infer_records_{select_model}", help="Each record must supply all model feature columns with the same semantics and units used in training.")
            if st.button("Run prediction", type="primary", key=f"run_prediction_{select_model}"):
                try:
                    records = json.loads(record_json)
                    if not isinstance(records, list) or not records or not all(isinstance(record, dict) for record in records):
                        raise ValueError("Provide a non-empty JSON array of objects.")
                except (json.JSONDecodeError, ValueError) as exc:
                    st.error(f"Invalid prediction input: {exc}")
                else:
                    response, error, _ = api_post(api_url, meta["endpoint"], {"records": records}, timeout=30.0)
                    if error:
                        show_api_error(error)
                    elif response:
                        st.markdown("**Prediction response**")
                        st.json(response)
                        if select_model in {"electricity_theft_detection", "water_leakage_detection"}:
                            st.info("This output is an indicator for investigation, not an automated accusation or physical diagnosis.")
        else:
            empty_state("Model artifact not installed", model.get("message", "Train a model and save its artifact plus metadata before inference."), "⌘")
            st.caption("Model paths and artifact metadata are documented in ml/models/README.md.")

    with model_tabs[2]:
        for key, meta in MODEL_META.items():
            model = model_statuses.get(key, {})
            with st.container(border=True):
                col1, col2 = st.columns([2.2, 1], vertical_alignment="center")
                with col1:
                    st.markdown(f"**{meta['icon']}  {meta['label']}**")
                    st.caption(model.get("message", "No status returned."))
                with col2:
                    st.markdown(status_pill("READY" if model.get("ready") else "NOT READY", "ok" if model.get("ready") else "warn"), unsafe_allow_html=True)
                a, b, c = st.columns(3)
                a.caption(f"Artifact exists: {'Yes' if model.get('artifact_exists') else 'No'}")
                b.caption(f"Metadata exists: {'Yes' if model.get('metadata_exists') else 'No'}")
                c.caption(f"Version: {model.get('version') or '—'}")
                if model.get("metrics"):
                    st.json(model["metrics"], expanded=False)

# -----------------------------------------------------------------------------
# Page: API & help
# -----------------------------------------------------------------------------
else:
    section_heading("API, configuration & runbook", "Useful links and operational notes for running the three-layer application locally.", "SYSTEM")
    a, b = st.columns([1, 1], gap="large")
    with a:
        with st.container(border=True):
            st.markdown("### Backend connection")
            st.write("The Streamlit frontend talks to FastAPI over HTTP. It does not import training scripts or load model files directly.")
            st.code(f"Backend URL\n{api_url}\n\nHealth\n{api_url}/health\n\nOpenAPI docs\n{api_url}/docs", language="text")
            st.link_button("Open FastAPI interactive docs ↗", f"{api_url}/docs", use_container_width=True)
    with b:
        with st.container(border=True):
            st.markdown("### Start the application")
            st.write("Run these commands from the extracted project root in separate terminals.")
            st.code("python -m pip install -r requirements.txt\nuvicorn backend.main:app --reload --port 8000", language="bash")
            st.code("streamlit run frontend/app.py", language="bash")
            st.caption("Set BACKEND_URL when the API runs on another host. Do not expose the development server publicly without authentication and deployment hardening.")

    section_heading("Service map", "Core endpoints consumed by this control room.", "API CONTRACT")
    endpoint_rows = [
        {"Method": "GET", "Endpoint": "/health", "Purpose": "Backend liveness"},
        {"Method": "GET", "Endpoint": "/api/v1/dashboard", "Purpose": "Dataset/model overview"},
        {"Method": "GET", "Endpoint": "/api/v1/pipelines", "Purpose": "Pipeline catalog and readiness"},
        {"Method": "GET", "Endpoint": "/api/v1/datasets", "Purpose": "Raw/processed file inventory"},
        {"Method": "GET", "Endpoint": "/api/v1/pipelines/{key}/inspect", "Purpose": "Sample schema inspection"},
        {"Method": "POST", "Endpoint": "/api/v1/pipelines/{key}/prepare", "Purpose": "Start data preparation"},
        {"Method": "POST", "Endpoint": "/api/v1/pipelines/{key}/train", "Purpose": "Explicit model training request"},
        {"Method": "GET", "Endpoint": "/api/v1/jobs/{job_id}", "Purpose": "Job status and logs"},
        {"Method": "GET", "Endpoint": "/api/v1/pipelines/{key}/results", "Purpose": "Saved model evaluation results"},
        {"Method": "GET", "Endpoint": "/api/v1/models/status", "Purpose": "Inference artifact readiness"},
    ]
    st.dataframe(pd.DataFrame(endpoint_rows), hide_index=True, use_container_width=True)
    st.warning("Security note: the pipeline API launches fixed scripts with validated parameters, but the prototype is intended for a trusted local environment. Add authentication, authorization, and deployment controls before exposing it to the internet.")
