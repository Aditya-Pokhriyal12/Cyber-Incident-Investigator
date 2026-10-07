"""
Feature Extractor
=================
Converts parsed log entries and raw DataFrames into feature vectors
suitable for both rule-based detection and ML inference.

HTTP log features (per-entry):
  - method_encoded       : int  GET=0, POST=1, PUT=2, DELETE=3, HEAD=4, OPTIONS=5,
                                PATCH=6, TRACE=7, CONNECT=8, other=9
  - path_depth           : int  number of '/' segments in URL
  - path_len             : int  total character length of the URL path
  - status_code          : int  HTTP response code
  - status_class         : int  2=2xx, 3=3xx, 4=4xx, 5=5xx
  - response_size        : int  bytes in response body
  - response_time        : float ms (0.0 if missing)
  - is_admin_path        : int  1 if path targets /admin /root /manager etc.
  - is_login_path        : int  1 if path targets /login /auth /signin etc.
  - is_suspicious_method : int  1 if method is DELETE/PUT/TRACE/CONNECT
  - is_server_error      : int  1 if status 5xx
  - is_client_error      : int  1 if status 4xx

Network flow features (per-row from CSV):
  - payload_byte_1 … payload_byte_N  (already numeric)
  - payload_mean   : mean of all payload bytes
  - payload_std    : std  of all payload bytes
  - payload_nonzero: count of non-zero bytes
  - src_port       : int
  - dst_port       : int
  - proto_encoded  : int (tcp=0, udp=1, icmp=2, other=3)
  - duration       : float
"""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import pandas as pd

from app.models.schemas import HTTPLogEntry, NetworkFlowEntry
from app.config import N_PAYLOAD_FEATURES

# ── HTTP feature helpers ──────────────────────────────────────────────────────

# Contiguous encoding with no gaps; fallback (unknown method) = 9
_METHOD_MAP: Dict[str, int] = {
    "GET": 0, "POST": 1, "PUT": 2, "DELETE": 3,
    "HEAD": 4, "OPTIONS": 5, "PATCH": 6, "TRACE": 7, "CONNECT": 8,
}
_SUSPICIOUS_METHODS = {"DELETE", "PUT", "TRACE", "CONNECT"}
_ADMIN_KEYWORDS     = {"/admin", "/root", "/manager", "/console", "/phpmyadmin"}
_LOGIN_KEYWORDS     = {"/login", "/signin", "/auth", "/wp-login"}


def extract_http_features(entry: HTTPLogEntry) -> Dict[str, Any]:
    """Return a flat feature dict for one HTTP log entry.

    All boolean-like fields are stored as int (0/1) so sklearn and
    pandas treat them as numeric consistently.
    """
    path_lower  = entry.path.lower()
    status      = entry.status_code
    status_cls  = status // 100          # 2, 3, 4, or 5

    return {
        # ── raw / metadata (non-numeric, excluded from ML) ──
        "ip":                   entry.ip,
        "timestamp":            entry.timestamp,
        "method":               entry.method,
        "path":                 entry.path,
        # ── numeric features used by ML ──────────────────────
        "status_code":          status,
        "status_class":         status_cls,
        "response_size":        entry.response_size,
        "response_time":        entry.response_time or 0.0,
        "method_encoded":       _METHOD_MAP.get(entry.method, 9),
        "path_depth":           entry.path.count("/"),
        "path_len":             len(entry.path),
        "is_admin_path":        int(any(kw in path_lower for kw in _ADMIN_KEYWORDS)),
        "is_login_path":        int(any(kw in path_lower for kw in _LOGIN_KEYWORDS)),
        "is_suspicious_method": int(entry.method in _SUSPICIOUS_METHODS),
        "is_server_error":      int(status_cls == 5),
        "is_client_error":      int(status_cls == 4),
    }


def label_http_entry(row: pd.Series) -> str:
    """
    Derive a binary label (ATTACK / BENIGN) for one HTTP log row.

    Called via df.apply(label_http_entry, axis=1) on the feature DataFrame.

    Attack heuristics (per-entry; designed for datasets where each IP is unique):
      - Suspicious method (DELETE/PUT/TRACE/CONNECT)
      - Targeting admin paths (/admin, /root …)
      - Failed POST to login path (status != 200)
      - Server error (5xx) with fast response (flood symptom)
    """
    is_attack = (
        row.get("is_suspicious_method", 0) == 1
        or row.get("is_admin_path", 0) == 1
        or (row.get("is_login_path", 0) == 1
            and row.get("method_encoded", 0) == 1      # POST=1
            and row.get("status_code", 200) != 200)
        or (row.get("is_server_error", 0) == 1
            and row.get("response_time", 9999) < 100)
    )
    return "ATTACK" if is_attack else "BENIGN"


def extract_http_features_batch(entries: List[HTTPLogEntry]) -> pd.DataFrame:
    """Convert a list of HTTPLogEntry objects into a feature DataFrame."""
    return pd.DataFrame([extract_http_features(e) for e in entries])


# ── Network flow feature helpers ──────────────────────────────────────────────

_PROTO_MAP: Dict[str, int] = {"tcp": 0, "udp": 1, "icmp": 2}


def _encode_proto(proto: str) -> int:
    return _PROTO_MAP.get(str(proto).lower(), 3)


def extract_network_features_from_entry(flow: NetworkFlowEntry) -> Dict[str, Any]:
    """Return a flat feature dict for one NetworkFlowEntry (API submission)."""
    payload = np.array(flow.payload_bytes, dtype=np.float32)

    # Pad / truncate to exactly N_PAYLOAD_FEATURES
    if len(payload) < N_PAYLOAD_FEATURES:
        payload = np.pad(payload, (0, N_PAYLOAD_FEATURES - len(payload)))
    else:
        payload = payload[:N_PAYLOAD_FEATURES]

    feat: Dict[str, Any] = {f"payload_byte_{i+1}": int(payload[i])
                             for i in range(N_PAYLOAD_FEATURES)}
    feat["payload_mean"]    = float(payload.mean())
    feat["payload_std"]     = float(payload.std())
    feat["payload_nonzero"] = int(np.count_nonzero(payload))
    feat["src_port"]        = flow.src_port
    feat["dst_port"]        = flow.dst_port
    feat["proto_encoded"]   = _encode_proto(flow.proto)
    feat["duration"]        = flow.duration
    return feat


def extract_network_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive aggregate features from a CSV-loaded DataFrame.

    Adds:  payload_mean, payload_std, payload_nonzero, proto_encoded
    Keeps: src_port, dst_port, duration, label (if present)
    """
    payload_cols = [f"payload_byte_{i}" for i in range(1, N_PAYLOAD_FEATURES + 1)]
    # Only use columns that actually exist
    existing_payload = [c for c in payload_cols if c in df.columns]
    payload_data = df[existing_payload].values.astype(np.float32)

    df = df.copy()
    df["payload_mean"]    = payload_data.mean(axis=1)
    df["payload_std"]     = payload_data.std(axis=1)
    df["payload_nonzero"] = (payload_data != 0).sum(axis=1)

    if "proto" in df.columns:
        df["proto_encoded"] = df["proto"].apply(_encode_proto)

    return df


def get_ml_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the ordered list of numeric feature columns to use for ML training/inference.
    Excludes label, dataset_source, and raw string columns.
    """
    exclude = {"label", "dataset_source", "proto"}
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
