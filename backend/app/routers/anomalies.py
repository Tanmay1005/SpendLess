import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas.transactions import AnomalyResponse, AnomalyResult

router = APIRouter(tags=["ml"])

Z_SCORE_THRESHOLD = 3.0


@router.get("/anomalies", response_model=AnomalyResponse)
async def detect_anomalies():
    from app.main import app_state

    transactions = app_state["transactions"]
    if not transactions:
        raise HTTPException(status_code=400, detail="No transactions to analyze. Upload a CSV first.")

    models = app_state["models"]
    if not models.get("anomaly_detector"):
        raise HTTPException(status_code=503, detail="Anomaly detection model not loaded")

    df = pd.DataFrame([{
        "date": t.date,
        "amount": t.amount,
    } for t in transactions])

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

    # Z-score on amount
    stats = models["anomaly_stats"]
    z_scores = np.abs((df["amount"].values - stats["amount_mean"]) / stats["amount_std"])

    results = []
    for i, txn in enumerate(transactions):
        is_iso_anomaly = iso_predictions[i] == -1
        is_z_anomaly = z_scores[i] > Z_SCORE_THRESHOLD
        is_anomaly = is_iso_anomaly or is_z_anomaly
        anomaly_score = float(-iso_scores[i])  # Higher = more anomalous

        txn.anomaly_score = round(anomaly_score, 4)
        txn.is_anomaly = is_anomaly

        if is_anomaly:
            results.append(AnomalyResult(
                transaction_id=txn.id,
                anomaly_score=round(anomaly_score, 4),
                is_anomaly=True,
                z_score=round(float(z_scores[i]), 4),
            ))

    return AnomalyResponse(
        message=f"Found {len(results)} anomalies out of {len(transactions)} transactions",
        results=results,
    )
