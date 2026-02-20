from datetime import datetime
from pydantic import BaseModel


CATEGORIES = [
    "Food",
    "Transport",
    "Entertainment",
    "Bills",
    "Shopping",
    "Healthcare",
    "Income",
    "Other",
]


class TransactionIn(BaseModel):
    date: datetime
    description: str
    amount: float
    merchant: str


class TransactionOut(BaseModel):
    id: int
    date: datetime
    description: str
    amount: float
    merchant: str
    category: str | None = None
    anomaly_score: float | None = None
    is_anomaly: bool = False


class CategoryResult(BaseModel):
    transaction_id: int
    category: str
    confidence: float


class AnomalyResult(BaseModel):
    transaction_id: int
    anomaly_score: float
    is_anomaly: bool
    z_score: float


class UploadResponse(BaseModel):
    message: str
    count: int
    transactions: list[TransactionOut]


class CategorizeResponse(BaseModel):
    message: str
    results: list[CategoryResult]


class AnomalyResponse(BaseModel):
    message: str
    results: list[AnomalyResult]
