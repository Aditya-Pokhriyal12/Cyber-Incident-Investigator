"""
Rule-Based Detection Engine
============================
Applies deterministic signature rules to:
  1. HTTPLogEntry feature dicts  (from logfiles.log)
  2. Network flow feature dicts  (from CSV / API submissions)

Rules implemented
-----------------
HTTP / Web logs:
  R01 – Brute-Force Login       : ≥N POST requests to login/auth paths from same IP in window
  R02 – Admin Path Enumeration  : ≥N requests to /admin paths from same IP in window
  R03 – HTTP Flood (DoS)        : ≥N total requests from same IP in window
  R04 – Error Storm (Scanning)  : ≥N 4xx responses from same IP in window
  R05 – Suspicious HTTP Method  : DELETE / PUT / TRACE / CONNECT to sensitive paths
  R06 – SQL Injection Signature : common SQLi patterns in URL path
  R07 – Path Traversal          : ../  sequences in path

Network flows (CSV / API):
  R08 – Suspicious Destination Port : dst_port in known attack-target set
  R09 – High Payload Activity       : payload_mean above threshold
  R10 – Known Attack Label          : label column already says non-BENIGN (training data audit)

Each rule returns a list of RuleAlert objects (empty = clean).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd

from app.config import (
    BRUTE_FORCE_THRESHOLD,
    FLOOD_THRESHOLD,
    SCAN_COUNT_THRESHOLD,
    SUSPICIOUS_DST_PORTS,
    HIGH_PAYLOAD_MEAN_THRESHOLD,
    SUSPICIOUS_METHODS,
)
from app.models.schemas import RuleAlert

# ── Regex patterns for injection / traversal ──────────────────────────────────
_SQLI_PATTERN = re.compile(
    r"('|--|;|%27|%3B|union\s+select|select\s+.*\s+from|insert\s+into|"
    r"drop\s+table|or\s+1\s*=\s*1|and\s+1\s*=\s*1|xp_cmdshell)",
    re.IGNORECASE,
)
_TRAVERSAL_PATTERN = re.compile(r"\.\./|\.\.\\|%2e%2e%2f|%2e%2e/|\.\.%2f", re.IGNORECASE)

_LOGIN_PATHS  = {"/login", "/signin", "/auth", "/wp-login", "/usr/login"}
_ADMIN_PATHS  = {"/admin", "/usr/admin", "/root", "/manager", "/console", "/phpmyadmin"}


# ─────────────────────────────────────────────────────────────────────────────
# HTTP / Web-log rules
# ─────────────────────────────────────────────────────────────────────────────

def _path_matches(path: str, path_set: set) -> bool:
    p = path.lower()
    return any(p.startswith(kw) for kw in path_set)


def run_http_rules(features: List[Dict[str, Any]]) -> Dict[int, List[RuleAlert]]:
    """
    Run all HTTP rules over a batch of feature dicts.

    Returns a dict keyed by entry index (int), each value being a list of
    RuleAlert objects triggered for that entry across the batch.
    """
    # Per-IP counters
    ip_total:        defaultdict = defaultdict(int)
    ip_login_post:   defaultdict = defaultdict(int)
    ip_admin:        defaultdict = defaultdict(int)
    ip_4xx:          defaultdict = defaultdict(int)

    # Per-entry alerts (index → list)
    entry_alerts: Dict[int, List[RuleAlert]] = defaultdict(list)

    for idx, feat in enumerate(features):
        ip     = feat.get("ip", "unknown")
        method = feat.get("method", "")
        path   = feat.get("path", "")
        status = feat.get("status_code", 0)

        ip_total[ip] += 1

        if method == "POST" and _path_matches(path, _LOGIN_PATHS):
            ip_login_post[ip] += 1

        if _path_matches(path, _ADMIN_PATHS):
            ip_admin[ip] += 1

        if 400 <= status < 500:
            ip_4xx[ip] += 1

        # ── R05: Suspicious HTTP method ───────────────────────────────────────
        if method in SUSPICIOUS_METHODS:
            entry_alerts[idx].append(RuleAlert(
                rule_name="R05_SUSPICIOUS_METHOD",
                description=f"Suspicious HTTP method '{method}' used on path '{path}'",
                matched_on={"method": method, "path": path, "status": status, "ip": ip},
            ))

        # ── R06: SQL Injection signature ──────────────────────────────────────
        if _SQLI_PATTERN.search(path):
            entry_alerts[idx].append(RuleAlert(
                rule_name="R06_SQL_INJECTION",
                description=f"Possible SQL injection pattern detected in path: {path}",
                matched_on={"path": path, "ip": ip},
            ))

        # ── R07: Path traversal ───────────────────────────────────────────────
        if _TRAVERSAL_PATTERN.search(path):
            entry_alerts[idx].append(RuleAlert(
                rule_name="R07_PATH_TRAVERSAL",
                description=f"Path traversal sequence detected: {path}",
                matched_on={"path": path, "ip": ip},
            ))

    # ── Aggregate per-IP threshold rules ─────────────────────────────────────
    # Attach threshold violations to every entry from that IP

    brute_ips  = {ip for ip, cnt in ip_login_post.items() if cnt >= BRUTE_FORCE_THRESHOLD}
    admin_ips  = {ip for ip, cnt in ip_admin.items()      if cnt >= BRUTE_FORCE_THRESHOLD}
    flood_ips  = {ip for ip, cnt in ip_total.items()      if cnt >= FLOOD_THRESHOLD}
    scan_ips   = {ip for ip, cnt in ip_4xx.items()        if cnt >= SCAN_COUNT_THRESHOLD}

    for idx, feat in enumerate(features):
        ip = feat.get("ip", "unknown")

        if ip in brute_ips:
            entry_alerts[idx].append(RuleAlert(
                rule_name="R01_BRUTE_FORCE_LOGIN",
                description=(
                    f"IP {ip} made {ip_login_post[ip]} POST requests to login paths "
                    f"(threshold: {BRUTE_FORCE_THRESHOLD})"
                ),
                matched_on={"ip": ip, "login_post_count": ip_login_post[ip]},
            ))

        if ip in admin_ips:
            entry_alerts[idx].append(RuleAlert(
                rule_name="R02_ADMIN_ENUMERATION",
                description=(
                    f"IP {ip} made {ip_admin[ip]} requests to admin paths "
                    f"(threshold: {BRUTE_FORCE_THRESHOLD})"
                ),
                matched_on={"ip": ip, "admin_request_count": ip_admin[ip]},
            ))

        if ip in flood_ips:
            entry_alerts[idx].append(RuleAlert(
                rule_name="R03_HTTP_FLOOD",
                description=(
                    f"IP {ip} sent {ip_total[ip]} total requests "
                    f"(threshold: {FLOOD_THRESHOLD})"
                ),
                matched_on={"ip": ip, "total_requests": ip_total[ip]},
            ))

        if ip in scan_ips:
            entry_alerts[idx].append(RuleAlert(
                rule_name="R04_ERROR_STORM_SCAN",
                description=(
                    f"IP {ip} received {ip_4xx[ip]} 4xx responses "
                    f"(threshold: {SCAN_COUNT_THRESHOLD})"
                ),
                matched_on={"ip": ip, "error_4xx_count": ip_4xx[ip]},
            ))

    return entry_alerts   # idx -> [RuleAlert, ...]


# ─────────────────────────────────────────────────────────────────────────────
# Network-flow rules
# ─────────────────────────────────────────────────────────────────────────────

def run_network_rules(features: List[Dict[str, Any]]) -> Dict[int, List[RuleAlert]]:
    """
    Run network-flow rules over a batch of feature dicts.
    Returns dict: row_index -> [RuleAlert, ...]
    """
    row_alerts: Dict[int, List[RuleAlert]] = defaultdict(list)

    for idx, feat in enumerate(features):
        dst_port     = feat.get("dst_port", 0)
        payload_mean = feat.get("payload_mean", 0.0)
        label        = feat.get("label", "")

        # ── R08: Suspicious destination port ─────────────────────────────────
        if int(dst_port) in SUSPICIOUS_DST_PORTS:
            row_alerts[idx].append(RuleAlert(
                rule_name="R08_SUSPICIOUS_DST_PORT",
                description=f"Traffic to high-risk destination port {dst_port}",
                matched_on={"dst_port": dst_port},
            ))

        # ── R09: High payload activity ────────────────────────────────────────
        if float(payload_mean) > HIGH_PAYLOAD_MEAN_THRESHOLD:
            row_alerts[idx].append(RuleAlert(
                rule_name="R09_HIGH_PAYLOAD_ACTIVITY",
                description=(
                    f"Mean payload byte value {payload_mean:.1f} exceeds "
                    f"threshold {HIGH_PAYLOAD_MEAN_THRESHOLD}"
                ),
                matched_on={"payload_mean": payload_mean},
            ))

        # ── R10: Known attack label (training-data audit) ──────────────────────
        if label and label.upper() not in ("BENIGN", "NORMAL", "", "NAN", "NONE"):
            row_alerts[idx].append(RuleAlert(
                rule_name="R10_KNOWN_ATTACK_LABEL",
                description=f"Record carries known attack label: '{label}'",
                matched_on={"label": label},
            ))

    return row_alerts


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run rules directly on a DataFrame (for CSV batch mode)
# ─────────────────────────────────────────────────────────────────────────────

def run_network_rules_on_df(df: pd.DataFrame) -> Dict[int, List[RuleAlert]]:
    """Run network rules on a feature-enriched DataFrame."""
    records = df.to_dict(orient="records")
    return run_network_rules(records)
