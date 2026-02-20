from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database (Phase 2)
    database_url: str = "postgresql://spendlens:dev@localhost:5432/spendlens"

    # Redis (Phase 3)
    redis_url: str = "redis://localhost:6379"

    # Gemini (Phase 3)
    gemini_api_key: str = ""

    # ML
    model_dir: str = "ml/models"

    # Drift detection (Phase 8)
    drift_check_interval_hours: int = 24
    drift_psi_threshold: float = 0.2
    retrain_min_samples: int = 500

    # General
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
