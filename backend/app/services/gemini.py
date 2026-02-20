import logging

import google.generativeai as genai

from app.config import settings

logger = logging.getLogger("spendlens")

_model = None


def init_gemini():
    global _model
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY not set — advice endpoint will be unavailable")
        return
    genai.configure(api_key=settings.gemini_api_key)
    _model = genai.GenerativeModel("gemini-2.0-flash")
    logger.info("Gemini Flash model initialized")


async def generate_advice(spending_summary: dict) -> str:
    """Send aggregated spending summary to Gemini and get budget advice."""
    if _model is None:
        raise RuntimeError("Gemini not configured. Set GEMINI_API_KEY.")

    prompt = f"""You are a helpful personal finance advisor. Based on the following spending summary,
provide 3-5 specific, actionable budget recommendations. Be concise and practical.

Spending Summary:
- Total spending: ${spending_summary['total_spending']:.2f}
- Total income: ${spending_summary['total_income']:.2f}
- Savings rate: {spending_summary['savings_rate']:.1f}%

Spending by category:
{_format_categories(spending_summary['by_category'])}

Top merchants by spend:
{_format_merchants(spending_summary['top_merchants'])}

{f"Anomalies detected: {spending_summary['anomaly_count']} unusual transactions" if spending_summary.get('anomaly_count') else ""}

Provide your advice in a clear, friendly tone. Focus on areas where the user could save money
or optimize their spending. If the savings rate is negative, flag this as urgent."""

    response = await _model.generate_content_async(prompt)
    return response.text


def _format_categories(categories: dict) -> str:
    lines = []
    for cat, data in sorted(categories.items(), key=lambda x: x[1]["total"], reverse=True):
        lines.append(f"  - {cat}: ${data['total']:.2f} ({data['pct']:.1f}% of spending, {data['count']} transactions)")
    return "\n".join(lines)


def _format_merchants(merchants: list) -> str:
    lines = []
    for m in merchants[:10]:
        lines.append(f"  - {m['merchant']}: ${m['total']:.2f} ({m['count']} transactions)")
    return "\n".join(lines)
