import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas.transactions import CategorizeResponse, CategoryResult
from app.utils.preprocessing import extract_features

router = APIRouter(tags=["ml"])


@router.post("/categorize", response_model=CategorizeResponse)
async def categorize_transactions():
    from app.main import app_state

    transactions = app_state["transactions"]
    if not transactions:
        raise HTTPException(status_code=400, detail="No transactions to categorize. Upload a CSV first.")

    models = app_state["models"]
    if not models.get("categorizer"):
        raise HTTPException(status_code=503, detail="Categorizer model not loaded")

    uncategorized = [t for t in transactions if t.category is None]
    if not uncategorized:
        return CategorizeResponse(message="All transactions already categorized", results=[])

    df = pd.DataFrame([{
        "date": t.date,
        "description": t.description,
        "amount": t.amount,
        "merchant": t.merchant,
    } for t in uncategorized])

    features, _ = extract_features(df, tfidf=models["tfidf"], fit=False)
    predictions = models["categorizer"].predict(features)
    probabilities = models["categorizer"].predict_proba(features)

    results = []
    for i, txn in enumerate(uncategorized):
        category = models["label_encoder"].inverse_transform([predictions[i]])[0]
        confidence = float(np.max(probabilities[i]))
        txn.category = category
        results.append(CategoryResult(
            transaction_id=txn.id,
            category=category,
            confidence=round(confidence, 4),
        ))

    return CategorizeResponse(
        message=f"Categorized {len(results)} transactions",
        results=results,
    )
