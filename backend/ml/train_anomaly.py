"""Train Isolation Forest anomaly detector on synthetic data."""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from faker import Faker
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

fake = Faker()
Faker.seed(42)
np.random.seed(42)


def generate_normal_transactions(n_samples: int = 1500) -> pd.DataFrame:
    """Generate normal-looking transactions."""
    records = []
    for _ in range(n_samples):
        amount = round(np.random.lognormal(mean=3.5, sigma=1.0), 2)
        date = fake.date_time_between(start_date="-1y", end_date="now")
        records.append({
            "date": date,
            "amount": amount,
            "hour": date.hour,
            "day_of_week": date.weekday(),
            "is_weekend": int(date.weekday() >= 5),
        })
    return pd.DataFrame(records)


def extract_anomaly_features(df: pd.DataFrame) -> np.ndarray:
    """Extract numeric features for anomaly detection."""
    dates = pd.to_datetime(df["date"])
    features = pd.DataFrame({
        "amount": df["amount"].astype(float),
        "amount_log": np.log1p(df["amount"].astype(float).abs()),
        "hour": dates.dt.hour,
        "day_of_week": dates.dt.dayofweek,
        "is_weekend": (dates.dt.dayofweek >= 5).astype(int),
        "day_of_month": dates.dt.day,
    })
    return features.values


def main():
    print("Generating normal transaction data...")
    df = generate_normal_transactions(1500)
    print(f"  Generated {len(df)} normal transactions")

    print("Extracting features...")
    features = extract_anomaly_features(df)
    print(f"  Feature matrix shape: {features.shape}")

    print("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features)

    scores = model.decision_function(features)
    predictions = model.predict(features)
    n_anomalies = (predictions == -1).sum()
    print(f"  Detected {n_anomalies} anomalies in training data ({n_anomalies/len(df)*100:.1f}%)")

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "isolation_forest.pkl"))

    # Save training stats for z-score calculation
    stats = {
        "amount_mean": float(df["amount"].mean()),
        "amount_std": float(df["amount"].std()),
    }
    joblib.dump(stats, os.path.join(model_dir, "anomaly_stats.pkl"))

    print(f"Models saved to {model_dir}/")
    print("  - isolation_forest.pkl")
    print("  - anomaly_stats.pkl")


if __name__ == "__main__":
    main()
