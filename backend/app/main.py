import logging
import os
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services.cache import close_cache, init_cache
from app.services.db import close_db, init_db
from app.services.gemini import init_gemini

logger = logging.getLogger("spendlens")

app_state: dict = {
    "models": {},
}


def load_models():
    """Load ML models from disk into memory."""
    model_dir = settings.model_dir
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_dir)

    try:
        app_state["models"]["categorizer"] = joblib.load(
            os.path.join(model_dir, "xgb_categorizer.pkl")
        )
        app_state["models"]["tfidf"] = joblib.load(
            os.path.join(model_dir, "tfidf_vectorizer.pkl")
        )
        app_state["models"]["label_encoder"] = joblib.load(
            os.path.join(model_dir, "label_encoder.pkl")
        )
        app_state["models"]["anomaly_detector"] = joblib.load(
            os.path.join(model_dir, "isolation_forest.pkl")
        )
        app_state["models"]["anomaly_stats"] = joblib.load(
            os.path.join(model_dir, "anomaly_stats.pkl")
        )
        logger.info("All ML models loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}. Run train_categorizer.py and train_anomaly.py first.")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=settings.log_level)
    logger.info("Starting SpendLens backend...")
    load_models()
    await init_db()
    await init_cache()
    init_gemini()
    yield
    await close_cache()
    await close_db()
    logger.info("Shutting down SpendLens backend...")


app = FastAPI(
    title="SpendLens API",
    description="Personal finance analytics with ML categorization and anomaly detection",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routers import advice, anomalies, categorize, health, upload

app.include_router(health.router)
app.include_router(upload.router)
app.include_router(categorize.router)
app.include_router(anomalies.router)
app.include_router(advice.router)
