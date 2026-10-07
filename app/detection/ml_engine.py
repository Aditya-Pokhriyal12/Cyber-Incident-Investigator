"""
ML-Based Detection Engine
==========================
Handles TWO separate classifiers:

  1. Network model  — trained on CICIDS2017 + UNSW payload CSVs
                      features: 1500 payload bytes + derived stats + flow metadata
                      saved to: trained_models/rf_model.joblib, xgb_model.joblib

  2. HTTP model     — trained on logfiles.log
                      features: method, path, status, response_size/time etc.
                      saved to: trained_models/http_rf_model.joblib, http_xgb_model.joblib

Each model type has its own scaler and feature-name list so inference
always aligns correctly to the training feature space.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from app.config import (
    RF_MODEL_FILE,
    XGB_MODEL_FILE,
    SCALER_FILE,
    LABEL_ENCODER_FILE,
    FEATURE_NAMES_FILE,
    TRAINED_MODELS_DIR,
    ML_ATTACK_CONFIDENCE_THRESHOLD,
    N_PAYLOAD_FEATURES,
)
from app.ingestion.log_reader import load_combined_dataset, load_http_dataset
from app.preprocessing.feature_extractor import (
    extract_network_features_from_df,
    get_ml_feature_columns,
)
from app.models.schemas import MLPrediction

logger = logging.getLogger(__name__)

# ── File paths for HTTP models ────────────────────────────────────────────────
HTTP_RF_MODEL_FILE      = os.path.join(TRAINED_MODELS_DIR, "http_rf_model.joblib")
HTTP_XGB_MODEL_FILE     = os.path.join(TRAINED_MODELS_DIR, "http_xgb_model.joblib")
HTTP_SCALER_FILE        = os.path.join(TRAINED_MODELS_DIR, "http_scaler.joblib")
HTTP_FEATURE_NAMES_FILE = os.path.join(TRAINED_MODELS_DIR, "http_feature_names.joblib")

# ── Module-level model cache ──────────────────────────────────────────────────
# Network models
_rf_model:       Optional[RandomForestClassifier] = None
_xgb_model:      Optional[XGBClassifier]          = None
_scaler:         Optional[StandardScaler]          = None
_label_encoder:  Optional[LabelEncoder]            = None
_feature_names:  Optional[List[str]]               = None

# HTTP models
_http_rf_model:      Optional[RandomForestClassifier] = None
_http_xgb_model:     Optional[XGBClassifier]          = None
_http_scaler:        Optional[StandardScaler]          = None
_http_feature_names: Optional[List[str]]               = None


def _binarize_label(label: str) -> int:
    """BENIGN/NORMAL → 0 ; everything else → 1."""
    return 0 if str(label).strip().upper() in ("BENIGN", "NORMAL") else 1


def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5,
        n_jobs=-1, random_state=42, class_weight="balanced",
    )


def _make_xgb(scale_pos_weight: float = 1.0) -> XGBClassifier:
    # use_label_encoder was removed in xgboost >= 1.6 — do NOT include it
    return XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )


def _fit_and_report(
    rf: RandomForestClassifier,
    xgb: XGBClassifier,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    rf_path: str,
    xgb_path: str,
) -> Dict[str, Any]:
    """Fit both models, save them, return metrics dict."""
    rf.fit(X_train, y_train)
    joblib.dump(rf, rf_path)
    y_rf = rf.predict(X_test)
    rf_rep = classification_report(y_test, y_rf, output_dict=True, zero_division=0)

    xgb.fit(X_train, y_train)
    joblib.dump(xgb, xgb_path)
    y_xgb = xgb.predict(X_test)
    xgb_rep = classification_report(y_test, y_xgb, output_dict=True, zero_division=0)

    return {
        "random_forest": {
            "accuracy":  rf_rep.get("accuracy", 0.0),
            "precision": rf_rep.get("1", {}).get("precision", 0.0),
            "recall":    rf_rep.get("1", {}).get("recall", 0.0),
            "f1":        rf_rep.get("1", {}).get("f1-score", 0.0),
        },
        "xgboost": {
            "accuracy":  xgb_rep.get("accuracy", 0.0),
            "precision": xgb_rep.get("1", {}).get("precision", 0.0),
            "recall":    xgb_rep.get("1", {}).get("recall", 0.0),
            "f1":        xgb_rep.get("1", {}).get("f1-score", 0.0),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_models(n_rows_each: int = 5000) -> Dict[str, Any]:
    """
    Train both the network-flow model and the HTTP-log model.

    Network model  → trained on CICIDS2017 + UNSW (n_rows_each rows each)
    HTTP model     → trained on logfiles.log (n_rows_each * 2 lines)

    All models are saved to trained_models/ and the in-memory cache is refreshed.
    """
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    results: Dict[str, Any] = {}

    # ── Network-flow model ────────────────────────────────────────────────────
    logger.info("Training NETWORK model on CICIDS2017 + UNSW …")
    df_net = load_combined_dataset(n_rows_each=n_rows_each)
    df_net = extract_network_features_from_df(df_net)
    y_net  = df_net["label"].apply(_binarize_label).values

    net_feature_cols = get_ml_feature_columns(df_net)
    joblib.dump(net_feature_cols, FEATURE_NAMES_FILE)

    scaler_net = StandardScaler()
    X_net = scaler_net.fit_transform(df_net[net_feature_cols].values.astype(np.float32))
    joblib.dump(scaler_net, SCALER_FILE)

    le = LabelEncoder()
    le.fit(df_net["label"].values)
    joblib.dump(le, LABEL_ENCODER_FILE)

    spw_net = float((y_net == 0).sum()) / max(float(y_net.sum()), 1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_net, y_net, test_size=0.2, random_state=42, stratify=y_net
    )
    net_metrics = _fit_and_report(
        _make_rf(), _make_xgb(spw_net),
        X_tr, X_te, y_tr, y_te,
        RF_MODEL_FILE, XGB_MODEL_FILE,
    )
    results["network"] = {
        "total_samples":  len(y_net),
        "attack_samples": int(y_net.sum()),
        "benign_samples": int((y_net == 0).sum()),
        "feature_count":  len(net_feature_cols),
        **net_metrics,
    }
    logger.info("Network RF accuracy: %.4f", net_metrics["random_forest"]["accuracy"])
    logger.info("Network XGB accuracy: %.4f", net_metrics["xgboost"]["accuracy"])

    # ── HTTP log model ────────────────────────────────────────────────────────
    logger.info("Training HTTP model on logfiles.log …")
    df_http = load_http_dataset(n_rows=n_rows_each * 2)
    y_http  = df_http["label"].apply(_binarize_label).values

    http_feature_cols = get_ml_feature_columns(df_http)
    joblib.dump(http_feature_cols, HTTP_FEATURE_NAMES_FILE)

    scaler_http = StandardScaler()
    X_http = scaler_http.fit_transform(df_http[http_feature_cols].values.astype(np.float32))
    joblib.dump(scaler_http, HTTP_SCALER_FILE)

    spw_http = float((y_http == 0).sum()) / max(float(y_http.sum()), 1)
    X_htr, X_hte, y_htr, y_hte = train_test_split(
        X_http, y_http, test_size=0.2, random_state=42, stratify=y_http
    )
    http_metrics = _fit_and_report(
        _make_rf(), _make_xgb(spw_http),
        X_htr, X_hte, y_htr, y_hte,
        HTTP_RF_MODEL_FILE, HTTP_XGB_MODEL_FILE,
    )
    results["http"] = {
        "total_samples":  len(y_http),
        "attack_samples": int(y_http.sum()),
        "benign_samples": int((y_http == 0).sum()),
        "feature_count":  len(http_feature_cols),
        **http_metrics,
    }
    logger.info("HTTP RF accuracy: %.4f", http_metrics["random_forest"]["accuracy"])
    logger.info("HTTP XGB accuracy: %.4f", http_metrics["xgboost"]["accuracy"])

    _load_models(force=True)
    _load_http_models(force=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_models(force: bool = False) -> bool:
    """Load persisted NETWORK models into cache. Returns True on success."""
    global _rf_model, _xgb_model, _scaler, _label_encoder, _feature_names

    if not force and _rf_model is not None:
        return True

    required = [RF_MODEL_FILE, XGB_MODEL_FILE, SCALER_FILE,
                LABEL_ENCODER_FILE, FEATURE_NAMES_FILE]
    if not all(os.path.exists(f) for f in required):
        logger.warning("Network model files not found. Run /ml/train first.")
        return False

    _rf_model      = joblib.load(RF_MODEL_FILE)
    _xgb_model     = joblib.load(XGB_MODEL_FILE)
    _scaler        = joblib.load(SCALER_FILE)
    _label_encoder = joblib.load(LABEL_ENCODER_FILE)
    _feature_names = joblib.load(FEATURE_NAMES_FILE)
    logger.info("Network models loaded from disk.")
    return True


def _load_http_models(force: bool = False) -> bool:
    """Load persisted HTTP models into cache. Returns True on success."""
    global _http_rf_model, _http_xgb_model, _http_scaler, _http_feature_names

    if not force and _http_rf_model is not None:
        return True

    required = [HTTP_RF_MODEL_FILE, HTTP_XGB_MODEL_FILE,
                HTTP_SCALER_FILE, HTTP_FEATURE_NAMES_FILE]
    if not all(os.path.exists(f) for f in required):
        logger.warning("HTTP model files not found. Run /ml/train first.")
        return False

    _http_rf_model      = joblib.load(HTTP_RF_MODEL_FILE)
    _http_xgb_model     = joblib.load(HTTP_XGB_MODEL_FILE)
    _http_scaler        = joblib.load(HTTP_SCALER_FILE)
    _http_feature_names = joblib.load(HTTP_FEATURE_NAMES_FILE)
    logger.info("HTTP models loaded from disk.")
    return True


def models_ready() -> bool:
    """Return True if NETWORK models are available."""
    return _load_models()


def http_models_ready() -> bool:
    """Return True if HTTP models are available."""
    return _load_http_models()


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers — shared
# ─────────────────────────────────────────────────────────────────────────────

def _build_vector(feat_dict: Dict[str, Any], feature_names: List[str]) -> np.ndarray:
    """Align a feature dict to the expected feature order, defaulting missing to 0."""
    return np.array(
        [float(feat_dict.get(col, 0)) for col in feature_names],
        dtype=np.float32,
    ).reshape(1, -1)


def _predict_with(
    rf, xgb, scaler, feature_names,
    feature_dicts: List[Dict[str, Any]],
) -> List[List[MLPrediction]]:
    """Generic batch inference for any (rf, xgb, scaler, feature_names) tuple."""
    if not feature_dicts:
        return []

    X = np.vstack([_build_vector(f, feature_names) for f in feature_dicts])
    X_scaled = scaler.transform(X)

    rf_probas  = rf.predict_proba(X_scaled)
    xgb_probas = xgb.predict_proba(X_scaled)

    results: List[List[MLPrediction]] = []
    for i in range(len(feature_dicts)):
        entry_preds: List[MLPrediction] = []
        for probas, name in [(rf_probas, "RandomForest"), (xgb_probas, "XGBoost")]:
            attack_prob = float(probas[i][1]) if probas.shape[1] > 1 else float(probas[i][0])
            is_attack   = attack_prob >= ML_ATTACK_CONFIDENCE_THRESHOLD
            entry_preds.append(MLPrediction(
                model=name,
                predicted_label="ATTACK" if is_attack else "BENIGN",
                confidence=round(attack_prob, 4),
                is_attack=is_attack,
            ))
        results.append(entry_preds)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Public inference API — Network
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(feature_dicts: List[Dict[str, Any]]) -> List[List[MLPrediction]]:
    """Run NETWORK model inference on a list of network-flow feature dicts."""
    if not _load_models():
        return [[] for _ in feature_dicts]
    return _predict_with(_rf_model, _xgb_model, _scaler, _feature_names, feature_dicts)


def predict_from_df(df: pd.DataFrame) -> List[List[MLPrediction]]:
    """Run NETWORK model inference on a feature-enriched DataFrame."""
    if not _load_models():
        return [[] for _ in range(len(df))]
    return predict_batch(df.to_dict(orient="records"))


def predict_single(feat_dict: Dict[str, Any]) -> List[MLPrediction]:
    """Run NETWORK model inference on a single feature dict."""
    if not _load_models():
        return []
    return predict_batch([feat_dict])[0]


# ─────────────────────────────────────────────────────────────────────────────
# Public inference API — HTTP
# ─────────────────────────────────────────────────────────────────────────────

def predict_http_batch(feature_dicts: List[Dict[str, Any]]) -> List[List[MLPrediction]]:
    """Run HTTP model inference on a list of HTTP-log feature dicts."""
    if not _load_http_models():
        return [[] for _ in feature_dicts]
    return _predict_with(
        _http_rf_model, _http_xgb_model,
        _http_scaler, _http_feature_names,
        feature_dicts,
    )


def predict_http_single(feat_dict: Dict[str, Any]) -> List[MLPrediction]:
    """Run HTTP model inference on a single feature dict."""
    if not _load_http_models():
        return []
    return predict_http_batch([feat_dict])[0]
