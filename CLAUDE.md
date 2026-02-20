# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpendLens — a containerized personal finance analytics platform. Ingests transaction CSVs, categorizes spending via XGBoost, detects anomalies with Isolation Forest + z-score, delivers budget advice via Google Gemini Flash, and self-monitors for data drift with automatic model retraining. Streamlit frontend, 7 Docker containers orchestrated via Compose.

## Architecture

Nginx (port 80) → routes to Streamlit UI (8501) and FastAPI backend (8000). Backend connects to Postgres (5432) for storage, Redis (6379) for caching, and Gemini Flash API for LLM advice. Prometheus (9090) + Grafana (3000) for monitoring. ML model artifacts live on a shared Docker volume (`model-artifacts`), not baked into images.

## Tech Stack

- **Backend:** Python 3.11, FastAPI, Uvicorn, asyncpg
- **ML:** scikit-learn, XGBoost (categorizer), Isolation Forest (anomaly detection)
- **MLOps:** PSI drift detection (scipy), APScheduler, model versioning in Postgres, hot-swap with threading.Lock
- **LLM:** Google Gemini Flash (external API, responses cached in Redis with 24h TTL)
- **Frontend:** Streamlit, Plotly
- **Database:** PostgreSQL 16
- **Cache:** Redis 7
- **Proxy:** Nginx
- **Monitoring:** Prometheus + Grafana
- **CI/CD:** GitHub Actions → Docker Hub → cloud VM

## Build & Run

```bash
# Train ML models (first time only)
cd backend/ml
python train_categorizer.py
python train_anomaly.py

# Build and run all services
docker-compose up --build

# Run backend tests
cd backend && pytest

# Lint
ruff check backend/
```

## Key Design Decisions

- Model `.pkl` files stored on Docker volume, not COPY'd into image (enables hot-swap retraining)
- Drift detection runs on INPUT features (amount distribution, merchant frequency, time patterns), NOT on model output categories
- Retraining uses only user-corrected labels via `POST /feedback`, not model predictions (avoids training on noise)
- Atomic model swap: load new model fully, swap pointer under threading.Lock
- Gemini receives only aggregated spending summaries, never raw transaction data (privacy)
- Single-user MVP (default user_id=1), auth added in final phase
- APScheduler stores last-check timestamp in Redis for crash recovery

## API Endpoints

Core:
- `POST /upload` — CSV transaction ingestion
- `POST /categorize` — ML classification
- `GET /anomalies` — flagged unusual transactions
- `POST /advice` — LLM budget recommendations
- `POST /feedback` — user corrects transaction categories (ground truth for retraining)
- `GET /health` — service health check
- `GET /metrics` — Prometheus scrape target

MLOps:
- `GET /drift/status` — latest drift detection results
- `GET /drift/models` — active model info and versions
- `POST /drift/check` — manually trigger drift check
- `POST /drift/retrain/{model_name}` — manually trigger retrain
- `POST /drift/reload` — hot-reload models from disk

## Phased Build Order

| Phase | What | Status |
|-------|------|--------|
| 1 | FastAPI + ML models (local, no Docker) | DONE |
| 2 | Postgres persistence | DONE |
| 3 | Redis caching + Gemini LLM advisor | NOT STARTED |
| 4 | Docker Compose (backend + postgres + redis) | NOT STARTED |
| 5 | Streamlit frontend | NOT STARTED |
| 6 | Nginx reverse proxy | NOT STARTED |
| 7 | Prometheus + Grafana monitoring | NOT STARTED |
| 8 | MLOps pipeline (drift + retrain + hot-swap) | NOT STARTED |
| 9 | CI/CD + auth + polish | NOT STARTED |

**Workflow:** After each phase is completed, stop, commit, and update this CLAUDE.md before starting the next phase. This preserves context across sessions.

## Environment Variables

Required: `GEMINI_API_KEY`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `REDIS_URL`
Optional: `DRIFT_CHECK_INTERVAL_HOURS` (def: 24), `DRIFT_PSI_THRESHOLD` (def: 0.2), `RETRAIN_MIN_SAMPLES` (def: 500), `LOG_LEVEL` (def: INFO)
