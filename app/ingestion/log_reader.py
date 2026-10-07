"""
Log Ingestion Module
====================
Handles reading from:
  1. logfiles.log  — Apache Combined Log Format
  2. CSV datasets  — Payload_data_CICIDS2017.csv / Payload_data_UNSW.csv

Exposes:
  - read_log_file(path)                    -> List[str]  (raw lines)
  - read_csv_flows(path, n_rows)           -> pd.DataFrame
  - stream_log_lines(path)                 -> Generator[str]
  - load_http_dataset(n_rows)              -> pd.DataFrame  (for HTTP ML training)
  - load_combined_dataset(n_rows_each)     -> pd.DataFrame  (network CSVs)
"""

from __future__ import annotations

import os
from typing import Generator, List, Optional

import pandas as pd

from app.config import (
    LOG_FILE_PATH,
    CICIDS_CSV_PATH,
    UNSW_CSV_PATH,
    N_PAYLOAD_FEATURES,
)


# ── HTTP log file reader ──────────────────────────────────────────────────────

def read_log_file(path: str = LOG_FILE_PATH) -> List[str]:
    """Read all lines from an Apache-style log file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Log file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def stream_log_lines(path: str = LOG_FILE_PATH) -> Generator[str, None, None]:
    """Yield log lines one at a time (memory-efficient for large files)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Log file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if stripped:
                yield stripped


# ── CSV / Network flow reader ─────────────────────────────────────────────────

def _payload_col_names() -> List[str]:
    """Return the list of payload_byte_1 … payload_byte_N column names."""
    return [f"payload_byte_{i}" for i in range(1, N_PAYLOAD_FEATURES + 1)]


def read_csv_flows(
    path: str,
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read a payload CSV (CICIDS2017 or UNSW format).

    Columns expected (last few):
        payload_byte_1 … payload_byte_1500, src_port, dst_port,
        proto, duration, label

    Returns a DataFrame with all columns present in the file.
    Missing payload columns are filled with 0.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, nrows=n_rows, low_memory=False)

    # Ensure all expected payload columns exist (fill missing with 0)
    payload_cols = _payload_col_names()
    existing = set(df.columns)
    missing = [c for c in payload_cols if c not in existing]
    if missing:
        df[missing] = 0

    # Coerce payload bytes to int16 to save memory
    df[payload_cols] = df[payload_cols].fillna(0).astype("int16")

    # Normalise label column name (strip whitespace)
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip()

    return df


def load_combined_dataset(n_rows_each: Optional[int] = None) -> pd.DataFrame:
    """
    Load and concatenate both CSV network datasets (CICIDS2017 + UNSW).
    Used for training the network-flow ML models.
    """
    df_cic  = read_csv_flows(CICIDS_CSV_PATH,  n_rows=n_rows_each)
    df_unsw = read_csv_flows(UNSW_CSV_PATH, n_rows=n_rows_each)
    df_cic["dataset_source"]  = "CICIDS2017"
    df_unsw["dataset_source"] = "UNSW"
    combined = pd.concat([df_cic, df_unsw], ignore_index=True)
    return combined


def load_http_dataset(n_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load logfiles.log, parse it, extract features, and derive labels
    for supervised HTTP ML training.

    Label derivation (per-entry, since each IP is unique in this dataset):
      ATTACK  if any of:
        - method is DELETE / PUT / TRACE / CONNECT  (suspicious method)
        - path contains /admin or /root             (admin targeting)
        - status_code is 500 (server error under attack)
        - path contains /login AND method is POST AND status != 200  (failed login attempt)
        - response_time < 100ms with status 5xx     (potential flood response)
      BENIGN  otherwise

    Returns a DataFrame with numeric feature columns + 'label'.
    """
    from app.preprocessing.log_parser import parse_log_lines
    from app.preprocessing.feature_extractor import (
        extract_http_features_batch,
        label_http_entry,
    )

    lines   = read_log_file(LOG_FILE_PATH)
    if n_rows:
        lines = lines[:n_rows]

    entries = parse_log_lines(lines)
    df      = extract_http_features_batch(entries)

    # Derive binary label from per-entry features
    df["label"] = df.apply(label_http_entry, axis=1)
    df["dataset_source"] = "HTTP_LOG"

    return df
