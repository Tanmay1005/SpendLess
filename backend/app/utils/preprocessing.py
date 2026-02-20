import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_features(df: pd.DataFrame, tfidf: TfidfVectorizer | None = None, fit: bool = False):
    """Extract features from transaction dataframe for ML models.

    Returns (feature_matrix, tfidf_vectorizer).
    Pass fit=True during training to fit the TF-IDF vectorizer.
    """
    # Text features from description + merchant
    text = (df["description"].fillna("") + " " + df["merchant"].fillna("")).str.lower()

    if fit:
        tfidf = TfidfVectorizer(max_features=100, stop_words="english")
        text_features = tfidf.fit_transform(text).toarray()
    else:
        text_features = tfidf.transform(text).toarray()

    # Date features
    dates = pd.to_datetime(df["date"])
    numeric_features = pd.DataFrame({
        "amount": df["amount"].astype(float),
        "day_of_week": dates.dt.dayofweek,
        "hour": dates.dt.hour,
        "day_of_month": dates.dt.day,
        "is_weekend": (dates.dt.dayofweek >= 5).astype(int),
        "amount_log": np.log1p(df["amount"].astype(float).abs()),
    }).values

    features = np.hstack([numeric_features, text_features])
    return features, tfidf
