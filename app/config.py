"""
Central configuration for the SIEM project.
All paths and thresholds live here so other modules import from one place.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data paths ───────────────────────────────────────────────────────────────
LOG_FILE_PATH         = os.path.join(BASE_DIR, "logfiles.log")
CICIDS_CSV_PATH       = os.path.join(BASE_DIR, "Payload_data_CICIDS2017.csv")
UNSW_CSV_PATH         = os.path.join(BASE_DIR, "Payload_data_UNSW.csv")
TRAINED_MODELS_DIR    = os.path.join(BASE_DIR, "trained_models")

# ── ML model filenames ────────────────────────────────────────────────────────
RF_MODEL_FILE         = os.path.join(TRAINED_MODELS_DIR, "rf_model.joblib")
XGB_MODEL_FILE        = os.path.join(TRAINED_MODELS_DIR, "xgb_model.joblib")
SCALER_FILE           = os.path.join(TRAINED_MODELS_DIR, "scaler.joblib")
LABEL_ENCODER_FILE    = os.path.join(TRAINED_MODELS_DIR, "label_encoder.joblib")
FEATURE_NAMES_FILE    = os.path.join(TRAINED_MODELS_DIR, "feature_names.joblib")

# ── Rule-based thresholds ─────────────────────────────────────────────────────
# HTTP brute-force: ≥ N requests to /login or /admin from one IP in window
BRUTE_FORCE_THRESHOLD = 10        # requests
BRUTE_FORCE_WINDOW_S  = 60        # seconds

# HTTP flood / DDoS: ≥ N requests from one IP in window
FLOOD_THRESHOLD       = 50
FLOOD_WINDOW_S        = 60

# Suspicious HTTP methods that should rarely appear on normal endpoints
SUSPICIOUS_METHODS    = {"DELETE", "PUT", "TRACE", "CONNECT"}

# Status codes that indicate scanning or enumeration
SCAN_STATUS_CODES     = {404, 400, 403}
SCAN_COUNT_THRESHOLD  = 20        # distinct 404/403/400 errors in window
SCAN_WINDOW_S         = 60

# Network: large payload byte mean (used on CSV data)
HIGH_PAYLOAD_MEAN_THRESHOLD = 200  # avg byte value

# Ports often targeted in scans
SUSPICIOUS_DST_PORTS  = {22, 23, 3389, 445, 135, 139, 1433, 3306, 5432}

# ── ML inference ─────────────────────────────────────────────────────────────
ML_ATTACK_CONFIDENCE_THRESHOLD = 0.60   # prob required to flag as attack
N_PAYLOAD_FEATURES             = 1500   # number of payload_byte_* columns used
