"""
SIEM Detection System – Main FastAPI Application
=================================================
Wires together all sub-modules:
  • Ingestion  (/ingest/*)
  • Detection  (/detect/*, /ml/*)

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ingestion.router import router as ingestion_router
from app.detection.router  import router as detection_router

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="SIEM Threat Detection System",
    description=(
        "A modular SIEM pipeline that ingests logs and network flows, "
        "applies rule-based detection and ML models (RandomForest + XGBoost) "
        "to identify cyber threats, and generates structured alerts with severity scoring."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS – open for development; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ingestion_router)
app.include_router(detection_router)


# ── Root / health ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "service": "SIEM Threat Detection System",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    from app.detection.ml_engine import models_ready
    import os
    from app.config import LOG_FILE_PATH, CICIDS_CSV_PATH, UNSW_CSV_PATH
    return {
        "api":          "ok",
        "models_ready": models_ready(),
        "data_sources": {
            "log_file":   os.path.exists(LOG_FILE_PATH),
            "cicids_csv": os.path.exists(CICIDS_CSV_PATH),
            "unsw_csv":   os.path.exists(UNSW_CSV_PATH),
        },
    }
