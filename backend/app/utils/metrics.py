from prometheus_client import Counter, Gauge, Histogram

# Request-level metrics are handled automatically by prometheus-fastapi-instrumentator.
# These are custom application-specific metrics.

# ML inference
INFERENCE_DURATION = Histogram(
    "spendlens_inference_duration_seconds",
    "Time spent on ML model inference",
    ["model"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

TRANSACTIONS_CATEGORIZED = Counter(
    "spendlens_transactions_categorized_total",
    "Total transactions categorized",
)

ANOMALIES_DETECTED = Counter(
    "spendlens_anomalies_detected_total",
    "Total anomalies detected",
)

TRANSACTIONS_UPLOADED = Counter(
    "spendlens_transactions_uploaded_total",
    "Total transactions uploaded",
)

# Cache
CACHE_HITS = Counter(
    "spendlens_cache_hits_total",
    "Cache hits",
    ["endpoint"],
)

CACHE_MISSES = Counter(
    "spendlens_cache_misses_total",
    "Cache misses",
    ["endpoint"],
)

# Gemini
GEMINI_REQUESTS = Counter(
    "spendlens_gemini_requests_total",
    "Total Gemini API requests",
    ["status"],
)

# Models
ACTIVE_MODEL_VERSION = Gauge(
    "spendlens_active_model_version",
    "Currently active model version",
    ["model"],
)
