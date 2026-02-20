"""Generate a sample CSV for testing the upload endpoint."""

import os
import sys

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(123)
np.random.seed(123)

MERCHANTS = {
    "Food": ["McDonald's", "Starbucks", "Chipotle", "Whole Foods", "Trader Joe's"],
    "Transport": ["Uber", "Lyft", "Shell", "Chevron", "Delta Airlines"],
    "Entertainment": ["Netflix", "Spotify", "AMC Theatres", "Steam", "Hulu"],
    "Bills": ["AT&T", "Verizon", "Comcast", "PG&E", "State Farm"],
    "Shopping": ["Amazon", "Walmart", "Target", "Best Buy", "Nike"],
    "Healthcare": ["CVS Pharmacy", "Walgreens", "Kaiser", "Dental Office", "Eye Center"],
    "Income": ["Employer Inc", "Freelance Client", "PayPal Transfer", "Direct Deposit", "Side Gig Co"],
    "Other": ["ATM Withdrawal", "Misc Store", "Local Shop", "Government Fee", "Wire Transfer"],
}

DESCRIPTIONS = {
    "Food": ["lunch at", "dinner at", "coffee from", "groceries at", "takeout from"],
    "Transport": ["ride to work", "gas fillup at", "flight booking", "parking at", "toll payment"],
    "Entertainment": ["monthly subscription", "movie night", "game purchase", "concert tickets", "streaming service"],
    "Bills": ["monthly bill", "phone payment", "internet service", "utility payment", "insurance premium"],
    "Shopping": ["online order from", "clothing from", "electronics from", "home goods at", "supplies from"],
    "Healthcare": ["prescription at", "doctor visit at", "pharmacy at", "dental checkup at", "eye exam at"],
    "Income": ["salary deposit", "freelance payment", "transfer received", "direct deposit", "bonus payment"],
    "Other": ["atm withdrawal", "miscellaneous charge", "service fee", "wire transfer", "donation to"],
}


def generate_sample_csv(n_rows: int = 500) -> pd.DataFrame:
    records = []
    for _ in range(n_rows):
        category = np.random.choice(list(MERCHANTS.keys()), p=[0.2, 0.15, 0.1, 0.15, 0.15, 0.1, 0.1, 0.05])
        merchant = np.random.choice(MERCHANTS[category])
        desc_template = np.random.choice(DESCRIPTIONS[category])
        description = f"{desc_template} {merchant}"

        if category == "Income":
            amount = -round(np.random.uniform(500, 5000), 2)
        else:
            amount = round(np.random.lognormal(mean=3.5, sigma=0.8), 2)

        date = fake.date_time_between(start_date="-6m", end_date="now")

        records.append({
            "date": date.strftime("%Y-%m-%d %H:%M:%S"),
            "description": description,
            "amount": amount,
            "merchant": merchant,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_sample_csv(500)
    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_transactions.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions → {output_path}")
