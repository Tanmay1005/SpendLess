"""Train XGBoost transaction categorizer on synthetic data."""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.utils.preprocessing import extract_features

fake = Faker()
Faker.seed(42)
np.random.seed(42)

CATEGORIES = {
    "Food": {
        "merchants": ["McDonald's", "Starbucks", "Chipotle", "Subway", "Pizza Hut",
                       "Whole Foods", "Trader Joe's", "Kroger", "Safeway", "Panera"],
        "descriptions": ["lunch", "dinner", "breakfast", "coffee", "groceries",
                         "takeout", "restaurant", "food delivery", "snacks", "meal"],
        "amount_range": (3.0, 150.0),
    },
    "Transport": {
        "merchants": ["Uber", "Lyft", "Shell", "Chevron", "BP", "ExxonMobil",
                       "Delta Airlines", "United Airlines", "Amtrak", "MTA"],
        "descriptions": ["gas", "fuel", "ride", "taxi", "parking", "toll",
                         "flight", "train ticket", "bus pass", "car wash"],
        "amount_range": (2.0, 500.0),
    },
    "Entertainment": {
        "merchants": ["Netflix", "Spotify", "AMC Theatres", "Steam", "PlayStation",
                       "Xbox", "Hulu", "Disney+", "Ticketmaster", "Barnes & Noble"],
        "descriptions": ["movie", "streaming", "concert", "game", "subscription",
                         "music", "books", "theater", "event tickets", "bowling"],
        "amount_range": (5.0, 200.0),
    },
    "Bills": {
        "merchants": ["AT&T", "Verizon", "Comcast", "PG&E", "State Farm",
                       "Allstate", "T-Mobile", "Water Utility", "Electric Co", "Sprint"],
        "descriptions": ["phone bill", "internet", "electricity", "water bill",
                         "insurance", "rent", "mortgage", "cable", "utility", "wifi"],
        "amount_range": (30.0, 2000.0),
    },
    "Shopping": {
        "merchants": ["Amazon", "Walmart", "Target", "Best Buy", "Nike",
                       "Zara", "H&M", "IKEA", "Home Depot", "Costco"],
        "descriptions": ["online order", "clothing", "electronics", "shoes",
                         "furniture", "home supplies", "gadget", "apparel", "decor", "tools"],
        "amount_range": (10.0, 500.0),
    },
    "Healthcare": {
        "merchants": ["CVS Pharmacy", "Walgreens", "Kaiser", "Blue Cross",
                       "UnitedHealth", "Quest Diagnostics", "LabCorp", "Rite Aid",
                       "Dental Office", "Eye Center"],
        "descriptions": ["prescription", "doctor visit", "pharmacy", "medical",
                         "dental", "eye exam", "therapy", "lab test", "copay", "health"],
        "amount_range": (10.0, 500.0),
    },
    "Income": {
        "merchants": ["Employer Inc", "Freelance Client", "PayPal Transfer",
                       "Direct Deposit", "Venmo", "Zelle", "Bank Transfer",
                       "Investment Div", "Tax Refund", "Side Gig Co"],
        "descriptions": ["salary", "paycheck", "freelance payment", "deposit",
                         "transfer received", "refund", "dividend", "bonus",
                         "reimbursement", "commission"],
        "amount_range": (100.0, 5000.0),
    },
    "Other": {
        "merchants": ["Misc Store", "Local Shop", "Service Provider", "Unknown",
                       "ATM Withdrawal", "Cash", "Wire Transfer", "Check",
                       "Government Fee", "Charity Org"],
        "descriptions": ["miscellaneous", "other", "fee", "charge", "withdrawal",
                         "donation", "gift", "service", "subscription", "payment"],
        "amount_range": (5.0, 300.0),
    },
}


def generate_synthetic_data(n_samples: int = 2000) -> pd.DataFrame:
    records = []
    samples_per_cat = n_samples // len(CATEGORIES)

    for category, config in CATEGORIES.items():
        for _ in range(samples_per_cat):
            merchant = np.random.choice(config["merchants"])
            description = np.random.choice(config["descriptions"])
            amount = round(np.random.uniform(*config["amount_range"]), 2)
            # Income is negative (credit)
            if category == "Income":
                amount = -amount
            date = fake.date_time_between(start_date="-1y", end_date="now")
            records.append({
                "date": date,
                "description": description,
                "amount": amount,
                "merchant": merchant,
                "category": category,
            })

    return pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)


def compute_training_distribution(df: pd.DataFrame) -> dict:
    """Compute input feature distributions for drift detection."""
    return {
        "amount_mean": float(df["amount"].mean()),
        "amount_std": float(df["amount"].std()),
        "amount_bins": np.histogram(df["amount"], bins=20)[0].tolist(),
        "amount_bin_edges": np.histogram(df["amount"], bins=20)[1].tolist(),
        "day_of_week_dist": pd.to_datetime(df["date"]).dt.dayofweek.value_counts(normalize=True).sort_index().tolist(),
        "merchant_counts": df["merchant"].value_counts().to_dict(),
        "n_samples": len(df),
    }


def main():
    print("Generating synthetic training data...")
    df = generate_synthetic_data(2000)
    print(f"  Generated {len(df)} transactions across {df['category'].nunique()} categories")

    print("Extracting features...")
    features, tfidf = extract_features(df, fit=True)
    print(f"  Feature matrix shape: {features.shape}")

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["category"])

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Training XGBoost categorizer...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "xgb_categorizer.pkl"))
    joblib.dump(tfidf, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))

    distribution = compute_training_distribution(df)
    with open(os.path.join(model_dir, "training_distribution.json"), "w") as f:
        json.dump(distribution, f, indent=2, default=str)

    print(f"Models saved to {model_dir}/")
    print("  - xgb_categorizer.pkl")
    print("  - tfidf_vectorizer.pkl")
    print("  - label_encoder.pkl")
    print("  - training_distribution.json")


if __name__ == "__main__":
    main()
