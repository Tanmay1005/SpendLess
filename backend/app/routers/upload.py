import io

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile

from app.schemas.transactions import TransactionOut, UploadResponse

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

    from app.main import app_state

    transactions = []
    start_id = len(app_state["transactions"]) + 1
    for i, row in df.iterrows():
        txn = TransactionOut(
            id=start_id + i,
            date=row["date"],
            description=str(row["description"]),
            amount=float(row["amount"]),
            merchant=str(row["merchant"]),
        )
        transactions.append(txn)

    app_state["transactions"].extend(transactions)

    return UploadResponse(
        message=f"Uploaded {len(transactions)} transactions",
        count=len(transactions),
        transactions=transactions,
    )
