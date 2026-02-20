import io

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile

from app.schemas.transactions import TransactionOut, UploadResponse
from app.services.db import get_pool

router = APIRouter(tags=["transactions"])


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    required_cols = {"date", "description", "amount", "merchant"}
    missing = required_cols - set(df.columns.str.lower())
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}",
        )

    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])

    pool = get_pool()
    transactions = []

    async with pool.acquire() as conn:
        for _, row in df.iterrows():
            record = await conn.fetchrow(
                """
                INSERT INTO transactions (user_id, date, description, amount, merchant)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, user_id, date, description, amount, merchant,
                          category, anomaly_score, is_anomaly
                """,
                1,  # default user_id for MVP
                row["date"].to_pydatetime(),
                str(row["description"]),
                float(row["amount"]),
                str(row["merchant"]),
            )
            transactions.append(TransactionOut(
                id=record["id"],
                date=record["date"],
                description=record["description"],
                amount=record["amount"],
                merchant=record["merchant"],
                category=record["category"],
                anomaly_score=record["anomaly_score"],
                is_anomaly=record["is_anomaly"],
            ))

    return UploadResponse(
        message=f"Uploaded {len(transactions)} transactions",
        count=len(transactions),
        transactions=transactions,
    )
