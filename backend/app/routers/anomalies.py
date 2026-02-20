import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas.transactions import AnomalyResponse, AnomalyResult
from app.services.db import get_pool

router = APIRouter(tags=["ml"])

Z_SCORE_THRESHOLD = 3.0


@router.get("/anomalies", response_model=AnomalyResponse)
async def detect_anomalies():
    from app.main import app_state

    models = app_state["models"]
    if not models.get("anomaly_detector"):
        raise HTTPException(status_code=503, detail="Anomaly detection model not loaded")

    pool = get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, date, amount
            FROM transactions
            WHERE user_id = $1
            ORDER BY id
            """,
            1,
        )

    if not rows:
        raise HTTPException(status_code=400, detail="No transactions to analyze. Upload a CSV first.")

    df = pd.DataFrame([dict(r) for r in rows])
    dates = pd.to_datetime(df["date"])
    features = pd.DataFrame({
        "amount": df["amount"].astype(float),
        "amount_log": np.log1p(df["amount"].astype(float).abs()),
        "hour": dates.dt.hour,
        "day_of_week": dates.dt.dayofweek,
        "is_weekend": (dates.dt.dayofweek >= 5).astype(int),
        "day_of_month": dates.dt.day,
    }).values

    iso_scores = models["anomaly_detector"].decision_function(features)
    iso_predictions = models["anomaly_detector"].predict(features)

    stats = models["anomaly_stats"]
    z_scores = np.abs((df["amount"].values - stats["amount_mean"]) / stats["amount_std"])

    results = []
    async with pool.acquire() as conn:
        for i, row in enumerate(rows):
            is_iso_anomaly = iso_predictions[i] == -1
            is_z_anomaly = z_scores[i] > Z_SCORE_THRESHOLD
            is_anomaly = bool(is_iso_anomaly or is_z_anomaly)
            anomaly_score = round(float(-iso_scores[i]), 4)

            await conn.execute(
                """
                UPDATE transactions SET anomaly_score = $1, is_anomaly = $2
                WHERE id = $3
                """,
                anomaly_score,
                is_anomaly,
                row["id"],
            )

            if is_anomaly:
                results.append(AnomalyResult(
                    transaction_id=row["id"],
                    anomaly_score=anomaly_score,
                    is_anomaly=True,
                    z_score=round(float(z_scores[i]), 4),
                ))

    return AnomalyResponse(
        message=f"Found {len(results)} anomalies out of {len(rows)} transactions",
        results=results,
    )
