# SpendLens — Intelligent Personal Finance Analyzer & Advisor

A production-grade, fully containerized personal finance analytics platform that classifies transactions using ML, detects anomalous spending, delivers personalized budget advice via Gemini Flash, and **self-monitors for data drift with automatic model retraining** — all served through a Streamlit dashboard.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│              Nginx (Reverse Proxy)                │
│              Port 80 / 443                        │
└─────────┬────────────────────┬───────────────────┘
          │                    │
┌─────────▼─────────┐  ┌──────▼───────────────────┐
│   Streamlit UI    │  │   FastAPI Backend         │
│   Port 8501       │  │   Port 8000               │
│                   │  │                            │
│ - Upload txns     │  │  POST /upload              │
│ - Spending dash   │  │  POST /categorize          │
│ - Anomaly alerts  │  │  GET  /anomalies           │
│ - Budget advice   │  │  POST /advice              │
│ - MLOps monitor   │  │  GET  /drift/status        │
│                   │  │  POST /drift/check         │
│                   │  │  POST /drift/retrain/{m}   │
│                   │  │  POST /drift/reload        │
│                   │  │  GET  /health               │
└───────────────────┘  └────┬──────────┬───────────┘
                            │          │
                     ┌──────▼──┐  ┌───▼────┐    ┌──────────────┐
                     │Postgres │  │ Redis  │    │ Gemini Flash │
                     │  :5432  │  │  :6379 │    │ (External)   │
                     │         │  │        │    └──────────────┘
                     │- Users  │  │- Cache │
                     │- Txns   │  │- Rate  │
                     │- Models │  │        │
                     │- Drift  │  │        │
                     └─────────┘  └────────┘

         ┌──────────────┐  ┌──────────────┐
         │  Prometheus  │  │   Grafana    │
         │    :9090     │  │    :3000     │
         └──────────────┘  └──────────────┘
```

**7 containers** orchestrated via Docker Compose. The only external dependency is Google Gemini Flash API for LLM-powered advice. All ML models are trained, versioned, and served locally inside Docker.

---

## Features

### Core Analytics
- **Transaction Categorization** — XGBoost classifier trained on synthetic bank data. Categories: Food, Transport, Entertainment, Bills, Shopping, Healthcare, Income, Other. Model weights served from a Docker volume.
- **Anomaly Detection** — Isolation Forest + z-score analysis flags unusual transactions based on user spending patterns.
- **LLM Budget Advisor** — Gemini Flash analyzes categorized spending summaries and returns personalized budget recommendations.
- **Streamlit Dashboard** — Upload CSV transactions, view categorized breakdowns, anomaly alerts, and AI-generated advice.

### MLOps (Dynamic Model Pipeline)
- **Data Drift Detection** — Scheduled PSI (Population Stability Index) checks compare recent transaction distributions against training data. Drift scores are logged to Postgres and exposed in Prometheus/Grafana.
- **Automatic Retraining** — When PSI exceeds the configured threshold, the retrain pipeline pulls recent labeled data from Postgres, retrains the model, and saves a new versioned `.pkl` to the Docker volume.
- **Model Versioning** — Every retrain produces a versioned artifact (`xgb_categorizer_v1.pkl`, `xgb_categorizer_v2.pkl`, etc.). The model registry in Postgres tracks version history, accuracy, and active status.
- **Hot-Swap Model Reload** — The `/drift/reload` endpoint reloads models from disk without restarting the container. Zero-downtime model updates.
- **MLOps Dashboard** — Streamlit page showing drift scores over time, model version history, retrain events, and current model health.

### Infrastructure
- **Caching** — Redis caches repeated queries and Gemini responses to reduce latency and API costs.
- **Monitoring** — Prometheus scrapes API metrics; Grafana dashboards track latency, error rates, cache hit ratio, model inference time, and drift scores.
- **CI/CD** — GitHub Actions pipeline builds, tests, and deploys on push to main.

---

## Tech Stack

| Layer              | Technology                        |
| ------------------ | --------------------------------- |
| Backend API        | Python 3.11, FastAPI, Uvicorn     |
| ML Models          | scikit-learn, XGBoost             |
| Drift Detection    | scipy (PSI, KS-test)             |
| LLM                | Google Gemini Flash (API)         |
| Scheduling         | APScheduler                       |
| Frontend           | Streamlit                         |
| Database           | PostgreSQL 16                     |
| Cache              | Redis 7                           |
| Reverse Proxy      | Nginx                             |
| Monitoring         | Prometheus + Grafana              |
| Containerization   | Docker, Docker Compose            |
| CI/CD              | GitHub Actions                    |
| Deployment         | GCP / AWS EC2 / DigitalOcean      |

---

## Project Structure

```
spendlens/
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py                 # FastAPI app + Prometheus metrics + lifespan
│   │   ├── config.py               # Settings & env var loading
│   │   ├── routers/
│   │   │   ├── upload.py           # POST /upload
│   │   │   ├── categorize.py       # POST /categorize
│   │   │   ├── anomalies.py        # GET  /anomalies
│   │   │   ├── advice.py           # POST /advice (Gemini Flash)
│   │   │   ├── drift.py            # Drift status, check, retrain, reload
│   │   │   └── health.py           # GET  /health
│   │   ├── services/
│   │   │   ├── model_manager.py    # Model loading, versioning, hot-swap
│   │   │   ├── drift_scheduler.py  # APScheduler periodic drift checks
│   │   │   ├── drift_detector.py   # PSI calculation logic
│   │   │   ├── retrainer.py        # Retrain pipeline (fetch data → train → save)
│   │   │   ├── gemini.py           # Gemini Flash API client
│   │   │   ├── cache.py            # Redis caching layer
│   │   │   └── db.py               # Postgres connection & queries
│   │   ├── schemas/
│   │   │   └── transactions.py     # Pydantic models
│   │   └── utils/
│   │       ├── metrics.py          # Prometheus metric helpers
│   │       └── preprocessing.py    # Feature engineering for ML
│   ├── ml/
│   │   ├── train_categorizer.py    # Initial training script
│   │   ├── train_anomaly.py        # Initial training script
│   │   └── artifacts/              # Docker volume mount point
│   │       ├── xgb_categorizer_v1.pkl
│   │       ├── isolation_forest_v1.pkl
│   │       └── training_distribution.json  # Baseline for drift detection
│   └── tests/
│       ├── test_categorize.py
│       ├── test_anomalies.py
│       ├── test_drift.py
│       └── test_advice.py
│
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py                      # Streamlit entry point
│   └── pages/
│       ├── 1_upload.py             # Transaction upload page
│       ├── 2_dashboard.py          # Spending breakdown & charts
│       ├── 3_anomalies.py          # Anomaly alerts
│       ├── 4_advice.py             # AI budget advisor
│       └── 5_mlops.py              # Drift scores, model versions, retrain history
│
├── nginx/
│   ├── Dockerfile
│   └── nginx.conf
│
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/prometheus.yml
│       │   └── dashboards/dashboards.yml
│       └── dashboards/
│           └── spendlens.json
│
├── db/
│   └── init.sql                    # Schema: users, transactions, model_registry, drift_log
│
├── data/
│   └── sample_transactions.csv
│
└── .github/
    └── workflows/
        └── deploy.yml
```

---

## Getting Started

### Prerequisites

- Docker & Docker Compose installed
- Google Gemini API key ([Get one here](https://aistudio.google.com/apikey))
- Git
- Python 3.11+ (for initial model training only)

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/spendlens.git
cd spendlens
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_key_here
POSTGRES_USER=spendlens
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=spendlens
REDIS_URL=redis://redis:6379/0
DRIFT_CHECK_INTERVAL_HOURS=24
DRIFT_PSI_THRESHOLD=0.2
RETRAIN_MIN_SAMPLES=500
```

> **⚠️ Never commit `.env` to Git.** It's already in `.gitignore`.

### 3. Train the initial ML models

```bash
cd backend/ml
python train_categorizer.py
python train_anomaly.py
```

This generates:
- `xgb_categorizer_v1.pkl` — initial categorizer
- `isolation_forest_v1.pkl` — initial anomaly detector
- `training_distribution.json` — baseline distribution for drift detection

These files are stored on a Docker volume, not baked into the image.

### 4. Build and run

```bash
docker-compose up --build
```

### 5. Access the app

| Service    | URL                          |
| ---------- | ---------------------------- |
| Streamlit  | http://localhost              |
| FastAPI    | http://localhost/api/docs     |
| Grafana    | http://localhost:3000         |
| Prometheus | http://localhost:9090         |

---

## How the Dynamic ML Pipeline Works

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│ User uploads │────▶│ Categorize with  │────▶│ Store in     │
│ new CSV      │     │ current model    │     │ Postgres     │
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                     │
                                                     ▼
                                           ┌──────────────────┐
                                           │ Scheduled Drift  │
                                           │ Check (every 24h)│
                                           │                  │
                                           │ PSI: recent txns │
                                           │ vs training dist │
                                           └────────┬─────────┘
                                                    │
                                         ┌──────────▼──────────┐
                                         │  PSI > threshold?   │
                                         └──────────┬──────────┘
                                              │            │
                                           YES ▼            ▼ NO
                                    ┌───────────────┐   Log & skip
                                    │ Retrain model │
                                    │ on recent data│
                                    └───────┬───────┘
                                            │
                                    ┌───────▼───────┐
                                    │ Save new .pkl │
                                    │ Update version│
                                    │ Hot-swap model│
                                    └───────────────┘
```

### Drift Detection Details

- **Metric:** Population Stability Index (PSI) computed on transaction amount distribution and category frequency distribution.
- **Baseline:** `training_distribution.json` saved alongside the initial model, recording the feature distributions used during training.
- **Schedule:** Runs every `DRIFT_CHECK_INTERVAL_HOURS` (default: 24h) via APScheduler.
- **Manual trigger:** `POST /drift/check` for on-demand drift evaluation.
- **Logging:** Every check is logged to the `drift_log` table and exposed as a Prometheus gauge (`spendlens_drift_psi_score`).

### Retrain Pipeline Details

- **Trigger:** Automatic (when drift detected) or manual (`POST /drift/retrain/{model_name}`).
- **Data source:** Recent transactions from Postgres (minimum `RETRAIN_MIN_SAMPLES` rows required).
- **Output:** New versioned `.pkl` saved to the Docker volume (e.g., `xgb_categorizer_v2.pkl`).
- **Registry:** `model_registry` table tracks every version with accuracy, F1 score, drift score at retrain time, and active/inactive status.
- **Hot-swap:** After saving, the backend reloads the new model in memory without container restart.

### Model Versioning

```
ml/artifacts/ (Docker volume)
├── xgb_categorizer_v1.pkl          # Initial model
├── xgb_categorizer_v2.pkl          # Retrained after drift
├── xgb_categorizer_v3.pkl          # Latest
├── isolation_forest_v1.pkl
├── isolation_forest_v2.pkl
└── training_distribution.json       # Updated after each retrain
```

The `model_registry` Postgres table serves as the source of truth:

| model_name      | version | is_active | accuracy | drift_score | created_at          |
| --------------- | ------- | --------- | -------- | ----------- | ------------------- |
| xgb_categorizer | 1       | false     | 0.87     | —           | 2025-06-01 10:00:00 |
| xgb_categorizer | 2       | false     | 0.89     | 0.24        | 2025-07-15 02:00:00 |
| xgb_categorizer | 3       | true      | 0.91     | 0.21        | 2025-08-20 02:00:00 |

---

## Docker Compose Services

| Service      | Image             | Port  | Purpose                                    |
| ------------ | ----------------- | ----- | ------------------------------------------ |
| `nginx`      | Custom            | 80    | Reverse proxy, routes traffic              |
| `frontend`   | Custom            | 8501  | Streamlit dashboard + MLOps monitor        |
| `backend`    | Custom            | 8000  | FastAPI + ML serving + drift detection     |
| `postgres`   | postgres:16       | 5432  | Transactions, model registry, drift log    |
| `redis`      | redis:7-alpine    | 6379  | Response caching & rate limiting           |
| `prometheus` | prom/prometheus   | 9090  | Metrics collection                         |
| `grafana`    | grafana/grafana   | 3000  | Monitoring dashboards                      |

**Key volume:** `model-artifacts` — shared Docker volume mounted to `/app/ml/artifacts` in the backend container. Models are read from and written to this volume, enabling persistence across container restarts and hot-swap without rebuilding images.

---

## ML Models

### Transaction Categorizer (XGBoost)

- **Input:** Transaction description, amount, merchant name, date features
- **Output:** Category label + confidence score
- **Training data:** Synthetic bank transaction dataset
- **Serving:** Loaded from Docker volume at startup. Reloaded on drift-triggered retrain.

### Anomaly Detector (Isolation Forest)

- **Input:** User's transaction history (amount, frequency, category patterns)
- **Output:** Anomaly score + binary flag per transaction
- **Logic:** Combines Isolation Forest with z-score thresholds on per-category spending

### LLM Advisor (Gemini Flash)

- **Input:** Categorized spending summary (aggregated, no raw transaction data sent)
- **Output:** Personalized budget recommendations
- **Note:** External API call. Responses cached in Redis (TTL: 24h).

---

## Monitoring

### Prometheus Metrics

| Metric                              | Type      | Description                          |
| ----------------------------------- | --------- | ------------------------------------ |
| `spendlens_requests_total`          | Counter   | Total API requests by endpoint       |
| `spendlens_request_latency_seconds` | Histogram | Request latency (p50/p95/p99)        |
| `spendlens_model_inference_seconds` | Histogram | Model inference time by model        |
| `spendlens_cache_hits_total`        | Counter   | Redis cache hits                     |
| `spendlens_cache_misses_total`      | Counter   | Redis cache misses                   |
| `spendlens_drift_psi_score`         | Gauge     | Latest PSI drift score by model      |
| `spendlens_active_model_version`    | Gauge     | Currently active model version       |

### Grafana Dashboards

Pre-configured dashboard includes:
- API latency (p50 / p95 / p99)
- Model inference time comparison
- Cache hit/miss ratio
- Drift PSI score over time with threshold line
- Retrain events as annotations
- Active model version indicator

Default Grafana credentials: `admin` / `admin`.

---

## API Endpoints

| Method | Endpoint               | Description                              |
| ------ | ---------------------- | ---------------------------------------- |
| POST   | `/upload`              | Upload CSV transaction file              |
| POST   | `/categorize`          | Categorize uploaded transactions         |
| GET    | `/anomalies`           | Get flagged anomalous transactions       |
| POST   | `/advice`              | Get LLM-powered budget recommendations   |
| GET    | `/drift/status`        | Latest drift detection results           |
| GET    | `/drift/models`        | Active model info and versions           |
| POST   | `/drift/check`         | Manually trigger drift check             |
| POST   | `/drift/retrain/{m}`   | Manually trigger model retrain           |
| POST   | `/drift/reload`        | Hot-reload models from disk              |
| GET    | `/health`              | Service health check                     |
| GET    | `/metrics`             | Prometheus metrics endpoint              |

---

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/deploy.yml`):

1. **On push to `main`:**
   - Run `pytest` test suite
   - Lint with `ruff`
   - Build Docker images
   - Push to Docker Hub / GitHub Container Registry
2. **On successful build:**
   - SSH into cloud VM
   - Pull latest images
   - `docker-compose up -d` with zero-downtime restart

---

## Deployment

### Cloud VM Setup

Minimum specs: **2 vCPU, 4GB RAM, 20GB disk**

```bash
ssh user@your-server-ip

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo apt install docker-compose-plugin

# Clone & configure
git clone https://github.com/<your-username>/spendlens.git
cd spendlens
cp .env.example .env
nano .env

# Run
docker-compose up -d
```

### Cost Estimate

| Item                          | Cost              |
| ----------------------------- | ----------------- |
| Docker, GitHub, Docker Hub    | Free              |
| Cloud VM (GCP free credits)   | Free for 3 months |
| Cloud VM (no free credits)    | $15–30/month      |
| Domain (optional)             | $10/year          |
| Gemini Flash API              | Free tier / ~$0   |

---

## Environment Variables

| Variable                      | Description                          | Required |
| ----------------------------- | ------------------------------------ | -------- |
| `GEMINI_API_KEY`              | Google Gemini Flash API key          | Yes      |
| `POSTGRES_USER`               | Postgres username                    | Yes      |
| `POSTGRES_PASSWORD`           | Postgres password                    | Yes      |
| `POSTGRES_DB`                 | Postgres database name               | Yes      |
| `REDIS_URL`                   | Redis connection string              | Yes      |
| `DRIFT_CHECK_INTERVAL_HOURS`  | Hours between drift checks (def: 24) | No       |
| `DRIFT_PSI_THRESHOLD`         | PSI threshold to trigger retrain     | No       |
| `RETRAIN_MIN_SAMPLES`         | Min rows required to retrain         | No       |
| `LOG_LEVEL`                   | Logging level (default: INFO)        | No       |

---

