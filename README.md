# SIEM Threat Detection System

A modular Security Information and Event Management (SIEM) pipeline built with **FastAPI**, **RandomForest**, and **XGBoost** that detects cyber threats from HTTP access logs and network flow data.

---

## Architecture

```
Log Sources (HTTP Logs / Network Flows / CSVs)
        ↓
Log Ingestion       — FastAPI endpoints to ingest logs & CSV data
        ↓
Parsing & Preprocessing — Apache log parser, feature extraction
        ↓
       ┌──────────────────┬──────────────────┐
Rule-Based Detection   ML-Based Detection
(10 signature rules)   (RandomForest + XGBoost)
       └──────────────────┴──────────────────┘
        ↓
Alert Generation    — Severity scoring (LOW / MEDIUM / HIGH / CRITICAL)
```

---

## Models

| Model | Dataset | Features |
|-------|---------|----------|
| Network RF + XGBoost | CICIDS2017 + UNSW-NB15 | 1500 payload bytes + flow metadata |
| HTTP RF + XGBoost | logfiles.log | Method, path, status, response time, etc. |

---

## Detection Rules

| Rule | Description |
|------|-------------|
| R01 | Brute Force Login — repeated POST to /login |
| R02 | Admin Path Enumeration |
| R03 | HTTP Flood / DoS |
| R04 | Error Storm Scanning (4xx threshold) |
| R05 | Suspicious HTTP Method (DELETE/PUT/TRACE) |
| R06 | SQL Injection pattern in URL |
| R07 | Path Traversal (../ sequences) |
| R08 | Suspicious Destination Port (22, 3389, 445…) |
| R09 | High Payload Activity |
| R10 | Known Attack Label (training data audit) |

---

## Project Structure

```
MajorProject/
├── app/
│   ├── config.py                   # Paths, thresholds, constants
│   ├── main.py                     # FastAPI app entry point
│   ├── models/schemas.py           # Pydantic models
│   ├── ingestion/
│   │   ├── log_reader.py           # Read log files + CSVs
│   │   └── router.py               # /ingest/* endpoints
│   ├── preprocessing/
│   │   ├── log_parser.py           # Apache log regex parser
│   │   └── feature_extractor.py   # Feature engineering
│   ├── detection/
│   │   ├── rule_engine.py          # 10 detection rules
│   │   ├── ml_engine.py            # RF + XGBoost train/predict
│   │   └── router.py               # /detect/* + /ml/* endpoints
│   └── alerts/
│       └── alert_generator.py      # Risk scoring + alert builder
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Usage

```bash
# Train models on all 3 datasets
curl -X POST "http://localhost:8000/ml/train?n_rows_each=5000"

# Check model status
curl "http://localhost:8000/ml/status"

# Detect threats in HTTP log file
curl "http://localhost:8000/detect/http/file?max_lines=500"

# Detect threats in CICIDS2017 network data
curl "http://localhost:8000/detect/network/cicids?n_rows=200"

# Detect threats in UNSW network data
curl "http://localhost:8000/detect/network/unsw?n_rows=200"
```

Browse the full interactive API docs at: `http://localhost:8000/docs`

---

## Datasets Required

Place these files in the project root (not included in repo due to size):

| File | Description |
|------|-------------|
| `logfiles.log` | Apache Combined Log Format — HTTP access logs |
| `Payload_data_CICIDS2017.csv` | CICIDS2017 network payload dataset |
| `Payload_data_UNSW.csv` | UNSW-NB15 network payload dataset |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET | `/ingest/status` | Data source availability |
| GET | `/ingest/logs/file` | Read raw log lines |
| GET | `/ingest/network/cicids` | Preview CICIDS2017 data |
| GET | `/ingest/network/unsw` | Preview UNSW data |
| POST | `/detect/http` | Detect threats in submitted log lines |
| GET | `/detect/http/file` | Detect threats in on-disk log file |
| POST | `/detect/network` | Detect threats in submitted flows |
| GET | `/detect/network/cicids` | Detect threats in CICIDS2017 |
| GET | `/detect/network/unsw` | Detect threats in UNSW |
| POST | `/ml/train` | Train all ML models |
| GET | `/ml/status` | Check model readiness |
