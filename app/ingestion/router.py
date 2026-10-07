"""
FastAPI router for log ingestion endpoints.

Endpoints
---------
GET  /ingest/logs/file          — ingest lines from the on-disk log file
POST /ingest/logs/raw           — submit raw log lines in the request body
GET  /ingest/network/cicids     — ingest rows from CICIDS2017 CSV
GET  /ingest/network/unsw       — ingest rows from UNSW CSV
POST /ingest/network/flows      — submit network flow records in request body
GET  /ingest/status             — health / source status check
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.config import LOG_FILE_PATH, CICIDS_CSV_PATH, UNSW_CSV_PATH
from app.ingestion.log_reader import (
    read_log_file,
    read_csv_flows,
    load_combined_dataset,
)
from app.models.schemas import (
    IngestLogRequest,
    IngestNetworkRequest,
    NetworkFlowEntry,
)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


# ── HTTP log endpoints ────────────────────────────────────────────────────────

@router.get("/logs/file", summary="Ingest lines from the on-disk log file")
def ingest_log_file(
    max_lines: Optional[int] = Query(None, description="Limit lines returned (default: all)")
):
    """Read the logfiles.log from disk and return raw lines."""
    try:
        lines = read_log_file(LOG_FILE_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if max_lines:
        lines = lines[:max_lines]

    return {
        "source": "logfiles.log",
        "total_lines": len(lines),
        "lines": lines,
    }


@router.post("/logs/raw", summary="Submit raw log lines in request body")
def ingest_raw_logs(payload: IngestLogRequest):
    """Accept a JSON list of raw log strings for downstream processing."""
    if not payload.lines:
        raise HTTPException(status_code=400, detail="No log lines provided")
    return {
        "source": "raw_post",
        "total_lines": len(payload.lines),
        "lines": payload.lines,
    }


# ── Network flow / CSV endpoints ──────────────────────────────────────────────

@router.get("/network/cicids", summary="Ingest rows from CICIDS2017 CSV")
def ingest_cicids(
    n_rows: Optional[int] = Query(100, description="Number of rows to load (default 100)")
):
    """Load N rows from the CICIDS2017 payload CSV and return as records."""
    try:
        df = read_csv_flows(CICIDS_CSV_PATH, n_rows=n_rows)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Return lightweight summary (not the full 1500-col payload)
    summary_cols = ["src_port", "dst_port", "proto", "duration", "label"]
    available = [c for c in summary_cols if c in df.columns]
    return {
        "source": "CICIDS2017",
        "total_rows": len(df),
        "columns": list(df.columns),
        "preview": df[available].head(10).to_dict(orient="records"),
    }


@router.get("/network/unsw", summary="Ingest rows from UNSW CSV")
def ingest_unsw(
    n_rows: Optional[int] = Query(100, description="Number of rows to load (default 100)")
):
    """Load N rows from the UNSW payload CSV and return as records."""
    try:
        df = read_csv_flows(UNSW_CSV_PATH, n_rows=n_rows)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    summary_cols = ["src_port", "dst_port", "proto", "duration", "label"]
    available = [c for c in summary_cols if c in df.columns]
    return {
        "source": "UNSW",
        "total_rows": len(df),
        "columns": list(df.columns),
        "preview": df[available].head(10).to_dict(orient="records"),
    }


@router.post("/network/flows", summary="Submit network flow records in request body")
def ingest_network_flows(payload: IngestNetworkRequest):
    """Accept a JSON list of NetworkFlowEntry objects for downstream processing."""
    if not payload.flows:
        raise HTTPException(status_code=400, detail="No flow records provided")
    return {
        "source": "raw_post",
        "total_flows": len(payload.flows),
        "flows": [f.model_dump() for f in payload.flows],
    }


# ── Health / status ───────────────────────────────────────────────────────────

@router.get("/status", summary="Check availability of all data sources")
def ingestion_status():
    """Quick health check confirming which data sources are accessible."""
    import os
    return {
        "log_file":    {"path": LOG_FILE_PATH,    "exists": os.path.exists(LOG_FILE_PATH)},
        "cicids_csv":  {"path": CICIDS_CSV_PATH,  "exists": os.path.exists(CICIDS_CSV_PATH)},
        "unsw_csv":    {"path": UNSW_CSV_PATH,    "exists": os.path.exists(UNSW_CSV_PATH)},
    }
