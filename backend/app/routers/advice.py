from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.cache import ADVICE_TTL, cache_get, cache_set, make_cache_key
from app.services.db import get_pool
from app.services.gemini import generate_advice
from app.utils.metrics import CACHE_HITS, CACHE_MISSES, GEMINI_REQUESTS

router = APIRouter(tags=["advice"])


class AdviceResponse(BaseModel):
    advice: str
    cached: bool = False


@router.post("/advice", response_model=AdviceResponse)
async def get_advice():
    pool = get_pool()

    # Build spending summary from DB
    async with pool.acquire() as conn:
        categorized_count = await conn.fetchval(
            "SELECT count(*) FROM transactions WHERE user_id = $1 AND category IS NOT NULL",
            1,
        )
        if not categorized_count:
            raise HTTPException(
                status_code=400,
                detail="No categorized transactions. Run /categorize first.",
            )

        # Spending by category (exclude Income)
        cat_rows = await conn.fetch(
            """
            SELECT category, count(*) as cnt, sum(amount) as total
            FROM transactions
            WHERE user_id = $1 AND category IS NOT NULL AND category != 'Income'
            GROUP BY category
            ORDER BY total DESC
            """,
            1,
        )

        # Total income
        income_row = await conn.fetchrow(
            """
            SELECT coalesce(sum(abs(amount)), 0) as total, count(*) as cnt
            FROM transactions
            WHERE user_id = $1 AND category = 'Income'
            """,
            1,
        )

        # Top merchants
        merchant_rows = await conn.fetch(
            """
            SELECT merchant, count(*) as cnt, sum(amount) as total
            FROM transactions
            WHERE user_id = $1 AND category != 'Income'
            GROUP BY merchant
            ORDER BY total DESC
            LIMIT 10
            """,
            1,
        )

        # Anomaly count
        anomaly_count = await conn.fetchval(
            "SELECT count(*) FROM transactions WHERE user_id = $1 AND is_anomaly = true",
            1,
        )

    total_spending = sum(r["total"] for r in cat_rows)
    total_income = float(income_row["total"]) if income_row else 0.0

    savings_rate = 0.0
    if total_income > 0:
        savings_rate = ((total_income - total_spending) / total_income) * 100

    by_category = {}
    for r in cat_rows:
        by_category[r["category"]] = {
            "total": float(r["total"]),
            "count": r["cnt"],
            "pct": (float(r["total"]) / total_spending * 100) if total_spending > 0 else 0,
        }

    spending_summary = {
        "total_spending": total_spending,
        "total_income": total_income,
        "savings_rate": savings_rate,
        "by_category": by_category,
        "top_merchants": [
            {"merchant": r["merchant"], "total": float(r["total"]), "count": r["cnt"]}
            for r in merchant_rows
        ],
        "anomaly_count": anomaly_count,
    }

    # Check cache
    cache_key = make_cache_key("advice", user_id=1, summary_hash=str(round(total_spending, 2)))
    cached = await cache_get(cache_key)
    if cached:
        CACHE_HITS.labels(endpoint="advice").inc()
        return AdviceResponse(advice=cached["advice"], cached=True)
    CACHE_MISSES.labels(endpoint="advice").inc()

    # Call Gemini
    try:
        advice_text = await generate_advice(spending_summary)
        GEMINI_REQUESTS.labels(status="success").inc()
    except RuntimeError as e:
        GEMINI_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        GEMINI_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")

    # Cache the response
    await cache_set(cache_key, {"advice": advice_text}, ttl=ADVICE_TTL)

    return AdviceResponse(advice=advice_text, cached=False)
