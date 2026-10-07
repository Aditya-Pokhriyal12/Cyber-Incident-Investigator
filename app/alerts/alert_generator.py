"""
Alert Generation Module
=======================
Combines rule-based and ML results into unified Alert objects.

Severity scoring
----------------
Base score starts at 0.

HTTP alerts:
  +30  if any rule fires
  +10  per additional rule that fires (capped)
  +20  if any ML model flags as attack (confidence-weighted)
  +15  if BOTH models agree it's an attack
  +10  for high-confidence ML (≥ 0.85)
  +10  for critical rules (SQLI, PATH_TRAVERSAL, BRUTE_FORCE)

Network alerts:
  +30  if any rule fires
  +15  per additional rule
  +20  if any ML model flags as attack
  +15  if BOTH models agree
  +10  for high-confidence ML (≥ 0.85)

Severity bands:
  0–25   → LOW
  26–50  → MEDIUM
  51–75  → HIGH
  76–100 → CRITICAL
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.models.schemas import (
    Alert,
    LogSource,
    MLPrediction,
    RuleAlert,
    Severity,
)

# ── Rules that escalate severity ──────────────────────────────────────────────
_CRITICAL_RULES = {
    "R01_BRUTE_FORCE_LOGIN",
    "R06_SQL_INJECTION",
    "R07_PATH_TRAVERSAL",
    "R08_SUSPICIOUS_DST_PORT",
    "R10_KNOWN_ATTACK_LABEL",
}


def _score_to_severity(score: float) -> Severity:
    if score <= 25:
        return Severity.LOW
    if score <= 50:
        return Severity.MEDIUM
    if score <= 75:
        return Severity.HIGH
    return Severity.CRITICAL


def _compute_risk_score(
    rule_alerts: List[RuleAlert],
    ml_predictions: List[MLPrediction],
) -> float:
    score = 0.0

    # Rule contribution
    if rule_alerts:
        score += 30.0
        score += min(10.0 * (len(rule_alerts) - 1), 20.0)   # up to +20 for extra rules
        for ra in rule_alerts:
            if ra.rule_name in _CRITICAL_RULES:
                score += 10.0

    # ML contribution
    attack_preds = [p for p in ml_predictions if p.is_attack]
    if attack_preds:
        score += 20.0
        # Confidence-weighted boost
        avg_conf = sum(p.confidence for p in attack_preds) / len(attack_preds)
        if avg_conf >= 0.85:
            score += 10.0
        if len(attack_preds) == len(ml_predictions) and len(ml_predictions) > 1:
            score += 15.0   # all models agree

    return min(score, 100.0)


def _attack_type_from_rules(rule_alerts: List[RuleAlert]) -> str:
    """Derive a human-readable attack type from the highest-priority rule."""
    priority = [
        ("R06_SQL_INJECTION",       "SQL Injection"),
        ("R07_PATH_TRAVERSAL",      "Path Traversal"),
        ("R01_BRUTE_FORCE_LOGIN",   "Brute Force Login"),
        ("R02_ADMIN_ENUMERATION",   "Admin Enumeration"),
        ("R03_HTTP_FLOOD",          "HTTP Flood / DoS"),
        ("R04_ERROR_STORM_SCAN",    "Port/Path Scan"),
        ("R05_SUSPICIOUS_METHOD",   "Suspicious HTTP Method"),
        ("R08_SUSPICIOUS_DST_PORT", "Network Port Scan"),
        ("R09_HIGH_PAYLOAD_ACTIVITY","High-Volume Data Transfer"),
        ("R10_KNOWN_ATTACK_LABEL",  "Known Attack (Labelled)"),
    ]
    rule_names = {ra.rule_name for ra in rule_alerts}
    for rule_id, label in priority:
        if rule_id in rule_names:
            return label
    return "Unknown Attack"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP alert builder
# ─────────────────────────────────────────────────────────────────────────────

def generate_http_alerts(
    entries: list,
    feature_dicts: List[Dict[str, Any]],
    rule_results: Dict[int, List[RuleAlert]],
    ml_results: List[List[MLPrediction]],
) -> List[Alert]:
    """
    Build Alert objects for HTTP log entries that have at least one
    rule match or at least one ML attack prediction.
    """
    alerts: List[Alert] = []

    for idx, (entry, feat) in enumerate(zip(entries, feature_dicts)):
        rule_alerts = rule_results.get(idx, [])
        ml_preds    = ml_results[idx] if idx < len(ml_results) else []

        # Only emit an alert if something fired
        has_rule  = len(rule_alerts) > 0
        has_ml    = any(p.is_attack for p in ml_preds)
        if not has_rule and not has_ml:
            continue

        risk_score = _compute_risk_score(rule_alerts, ml_preds)
        severity   = _score_to_severity(risk_score)
        attack_type = _attack_type_from_rules(rule_alerts) if has_rule else "ML-Detected Anomaly"

        desc_parts = []
        if has_rule:
            desc_parts.append(f"{len(rule_alerts)} rule(s) fired: "
                              + ", ".join(r.rule_name for r in rule_alerts))
        if has_ml:
            attack_models = [p.model for p in ml_preds if p.is_attack]
            desc_parts.append(f"ML flagged by: {', '.join(attack_models)}")

        alerts.append(Alert(
            alert_id=str(uuid.uuid4()),
            source=LogSource.HTTP_LOG,
            severity=severity,
            attack_type=attack_type,
            description=" | ".join(desc_parts),
            raw_entry=feat,
            rule_alerts=rule_alerts,
            ml_predictions=ml_preds,
            risk_score=round(risk_score, 2),
            timestamp=_now_iso(),
        ))

    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Network alert builder
# ─────────────────────────────────────────────────────────────────────────────

def generate_network_alerts(
    feature_dicts: List[Dict[str, Any]],
    rule_results: Dict[int, List[RuleAlert]],
    ml_results: List[List[MLPrediction]],
    source: str = "NETWORK",
) -> List[Alert]:
    """
    Build Alert objects for network flow entries that triggered at least
    one rule or one ML attack prediction.
    """
    alerts: List[Alert] = []

    for idx, feat in enumerate(feature_dicts):
        rule_alerts = rule_results.get(idx, [])
        ml_preds    = ml_results[idx] if idx < len(ml_results) else []

        has_rule = len(rule_alerts) > 0
        has_ml   = any(p.is_attack for p in ml_preds)
        if not has_rule and not has_ml:
            continue

        risk_score  = _compute_risk_score(rule_alerts, ml_preds)
        severity    = _score_to_severity(risk_score)
        attack_type = _attack_type_from_rules(rule_alerts) if has_rule else "ML-Detected Network Anomaly"

        desc_parts = []
        if has_rule:
            desc_parts.append(f"{len(rule_alerts)} rule(s) fired: "
                              + ", ".join(r.rule_name for r in rule_alerts))
        if has_ml:
            attack_models = [p.model for p in ml_preds if p.is_attack]
            desc_parts.append(f"ML flagged by: {', '.join(attack_models)}")

        # Lightweight raw entry (skip the 1500 payload columns)
        slim_feat = {k: v for k, v in feat.items()
                     if not k.startswith("payload_byte_")}

        alerts.append(Alert(
            alert_id=str(uuid.uuid4()),
            source=LogSource.NETWORK,
            severity=severity,
            attack_type=attack_type,
            description=" | ".join(desc_parts),
            raw_entry=slim_feat,
            rule_alerts=rule_alerts,
            ml_predictions=ml_preds,
            risk_score=round(risk_score, 2),
            timestamp=_now_iso(),
        ))

    return alerts
