"""
Pydantic schemas shared across the entire application.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# ── Severity levels ───────────────────────────────────────────────────────────
class Severity(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


# ── Source type ───────────────────────────────────────────────────────────────
class LogSource(str, Enum):
    HTTP_LOG = "HTTP_LOG"
    NETWORK  = "NETWORK"


# ── Raw log entry (parsed from logfiles.log) ──────────────────────────────────
class HTTPLogEntry(BaseModel):
    ip: str
    timestamp: str
    method: str
    path: str
    status_code: int
    response_size: int
    referrer: Optional[str] = None
    user_agent: Optional[str] = None
    response_time: Optional[float] = None


# ── Network flow entry (parsed from CSV datasets) ─────────────────────────────
class NetworkFlowEntry(BaseModel):
    src_port: int
    dst_port: int
    proto: str
    duration: float
    payload_bytes: List[int] = Field(..., description="List of up to 1500 payload byte values")
    label: Optional[str] = None          # present in training data, absent at inference


# ── Detection result from rule engine ────────────────────────────────────────
class RuleAlert(BaseModel):
    rule_name: str
    description: str
    matched_on: Dict[str, Any]


# ── Detection result from ML engine ──────────────────────────────────────────
class MLPrediction(BaseModel):
    model: str                          # "RandomForest" | "XGBoost"
    predicted_label: str
    confidence: float
    is_attack: bool


# ── Unified alert produced by alert generation ────────────────────────────────
class Alert(BaseModel):
    alert_id: str
    source: LogSource
    severity: Severity
    attack_type: str
    description: str
    raw_entry: Dict[str, Any]
    rule_alerts: List[RuleAlert] = []
    ml_predictions: List[MLPrediction] = []
    risk_score: float = Field(ge=0.0, le=100.0)
    timestamp: str


# ── API request / response wrappers ──────────────────────────────────────────
class IngestLogRequest(BaseModel):
    """POST body for submitting raw log lines."""
    lines: List[str] = Field(..., description="Raw log lines to ingest")


class IngestNetworkRequest(BaseModel):
    """POST body for submitting network flow records."""
    flows: List[NetworkFlowEntry]


class DetectionResponse(BaseModel):
    total_entries: int
    alerts_generated: int
    alerts: List[Alert]
