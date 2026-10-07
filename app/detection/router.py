"""
FastAPI router for detection endpoints.

Endpoints
---------
POST /detect/http               – run rules + ML on submitted log lines
GET  /detect/http/file          – run rules + ML on the on-disk log file
POST /detect/network            – run rules + ML on submitted flow records
GET  /detect/network/cicids     – run rules + ML on CICIDS2017 CSV
GET  /detect/network/unsw       – run rules + ML on UNSW CSV
POST /ml/train                  – trigger model training
GET  /ml/status                 – check whether models are ready
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.ingestion.log_reader import read_log_file, read_csv_flows
from app.preprocessing.log_parser import parse_log_lines
from app.preprocessing.feature_extractor import (
    extract_http_features_batch,
    extract_network_features_from_df,
    extract_network_features_from_entry,
)
from app.detection.rule_engine import run_http_rules, run_network_rules, run_network_rules_on_df
from app.detection.ml_engine import (
    train_models,
    models_ready,
    http_models_ready,
    predict_batch,
    predict_from_df,
    predict_http_batch,
)
from app.alerts.alert_generator import (
    generate_http_alerts,
    generate_network_alerts,
)
from app.models.schemas import (
    IngestLogRequest,
    IngestNetworkRequest,
    DetectionResponse,
)
from app.config import CICIDS_CSV_PATH, UNSW_CSV_PATH, LOG_FILE_PATH

router = APIRouter(tags=["Detection"])


# ─────────────────────────────────────────────────────────────────────────────
# HTTP / Web-log detection
# ─────────────────────────────────────────────────────────────────────────────

def _run_http_pipeline(raw_lines: list[str]) -> DetectionResponse:
    """Shared pipeline: parse → features → rules → HTTP ML → alerts."""
    entries = parse_log_lines(raw_lines)
    if not entries:
        raise HTTPException(status_code=422, detail="No valid log lines could be parsed.")

    feature_dicts = extract_http_features_batch(entries).to_dict(orient="records")

    rule_results = run_http_rules(feature_dicts)
    # Use the dedicated HTTP model — it was trained on HTTP features (method, path, status, etc.)
    ml_results = predict_http_batch(feature_dicts) if http_models_ready() else [[] for _ in feature_dicts]

    alerts = generate_http_alerts(entries, feature_dicts, rule_results, ml_results)

    return DetectionResponse(
        total_entries=len(entries),
        alerts_generated=len(alerts),
        alerts=alerts,
    )


@router.post("/detect/http", summary="Detect threats in submitted log lines")
def detect_http(payload: IngestLogRequest):
    return _run_http_pipeline(payload.lines)


@router.get("/detect/http/file", summary="Detect threats in the on-disk log file")
def detect_http_file(
    max_lines: Optional[int] = Query(None, description="Limit lines processed")
):
    try:
        lines = read_log_file(LOG_FILE_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if max_lines:
        lines = lines[:max_lines]

    return _run_http_pipeline(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Network-flow detection
# ─────────────────────────────────────────────────────────────────────────────

def _run_network_pipeline_from_df(df, source: str) -> DetectionResponse:
    """Shared pipeline for CSV-sourced network data."""
    df = extract_network_features_from_df(df)
    feature_dicts = df.to_dict(orient="records")

    rule_results = run_network_rules_on_df(df)
    ml_results   = predict_from_df(df) if models_ready() else [[] for _ in range(len(df))]

    alerts = generate_network_alerts(feature_dicts, rule_results, ml_results, source=source)

    return DetectionResponse(
        total_entries=len(df),
        alerts_generated=len(alerts),
        alerts=alerts,
    )


@router.post("/detect/network", summary="Detect threats in submitted flow records")
def detect_network(payload: IngestNetworkRequest):
    feature_dicts = [extract_network_features_from_entry(f) for f in payload.flows]

    rule_results = run_network_rules(feature_dicts)
    ml_results   = predict_batch(feature_dicts) if models_ready() else [[] for _ in feature_dicts]

    alerts = generate_network_alerts(feature_dicts, rule_results, ml_results, source="api_post")

    return DetectionResponse(
        total_entries=len(feature_dicts),
        alerts_generated=len(alerts),
        alerts=alerts,
    )


@router.get("/detect/network/cicids", summary="Detect threats in CICIDS2017 CSV")
def detect_cicids(
    n_rows: Optional[int] = Query(200, description="Rows to process (default 200)")
):
    try:
        df = read_csv_flows(CICIDS_CSV_PATH, n_rows=n_rows)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _run_network_pipeline_from_df(df, source="CICIDS2017")


@router.get("/detect/network/unsw", summary="Detect threats in UNSW CSV")
def detect_unsw(
    n_rows: Optional[int] = Query(200, description="Rows to process (default 200)")
):
    try:
        df = read_csv_flows(UNSW_CSV_PATH, n_rows=n_rows)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _run_network_pipeline_from_df(df, source="UNSW")


# ─────────────────────────────────────────────────────────────────────────────
# ML management
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/ml/train", summary="Train RandomForest + XGBoost on the dataset")
def ml_train(
    n_rows_each: int = Query(5000, description="Rows per dataset CSV (default 5000)")
):
    """
    Triggers full model training. This may take a minute depending on n_rows_each.
    Trained models are saved to trained_models/ for subsequent inference.
    """
    try:
        metrics = train_models(n_rows_each=n_rows_each)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")
    return {"status": "trained", "metrics": metrics}


@router.get("/ml/status", summary="Check whether ML models are loaded and ready")
def ml_status():
    net_ready  = models_ready()
    http_ready = http_models_ready()
    return {
        "network_models_ready": net_ready,
        "http_models_ready":    http_ready,
        "all_ready":            net_ready and http_ready,
        "message": (
            "All models loaded and ready for inference."
            if net_ready and http_ready
            else "Some models not found. POST /ml/train to train all models."
        ),
    }
