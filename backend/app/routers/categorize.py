import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas.transactions import CategorizeResponse, CategoryResult
from app.services.cache import cache_invalidate
from app.services.db import get_pool
from app.utils.metrics import INFERENCE_DURATION, TRANSACTIONS_CATEGORIZED
from app.utils.preprocessing import extract_features

router = APIRouter(tags=["ml"])


@router.post("/categorize", response_model=CategorizeResponse)
async def categorize_transactions():
    from app.main import app_state

    models = app_state["models"]
    if not models.get("categorizer"):
        raise HTTPException(status_code=503, detail="Categorizer model not loaded")

    pool = get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, date, description, amount, merchant
            FROM transactions
            WHERE user_id = $1 AND category IS NULL
            ORDER BY id
            """,
            1,
        )

    if not rows:
        return CategorizeResponse(message="All transactions already categorized", results=[])

    df = pd.DataFrame([dict(r) for r in rows])
    features, _ = extract_features(df, tfidf=models["tfidf"], fit=False)
    start = time.perf_counter()
    predictions = models["categorizer"].predict(features)
    probabilities = models["categorizer"].predict_proba(features)
    INFERENCE_DURATION.labels(model="categorizer").observe(time.perf_counter() - start)

    results = []
    async with pool.acquire() as conn:
        for i, row in enumerate(rows):
            category = models["label_encoder"].inverse_transform([predictions[i]])[0]
            confidence = float(np.max(probabilities[i]))

            await conn.execute(
                """
                UPDATE transactions SET category = $1, confidence = $2
                WHERE id = $3
                """,
                category,
                round(confidence, 4),
                row["id"],
            )

            results.append(CategoryResult(
                transaction_id=row["id"],
                category=category,
                confidence=round(confidence, 4),
            ))

    TRANSACTIONS_CATEGORIZED.inc(len(results))

    # Invalidate advice cache since categories changed
    await cache_invalidate("spendlens:advice:*")

    return CategorizeResponse(
        message=f"Categorized {len(results)} transactions",
        results=results,
    )
