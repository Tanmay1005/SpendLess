import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="SpendLens", page_icon="$", layout="wide")

# --- Sidebar Navigation ---
page = st.sidebar.radio(
    "Navigation",
    ["Upload Transactions", "Dashboard", "Anomalies", "AI Advice"],
)

# --- Session state defaults ---
if "transactions" not in st.session_state:
    st.session_state.transactions = None
if "anomalies" not in st.session_state:
    st.session_state.anomalies = None
if "advice" not in st.session_state:
    st.session_state.advice = None


def api(method: str, path: str, **kwargs):
    """Make a request to the backend API."""
    url = f"{BACKEND_URL}{path}"
    try:
        resp = requests.request(method, url, timeout=60, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Is the backend service running?")
        return None
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            detail = e.response.text
        st.error(f"API error: {e.response.status_code} — {detail}")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None


# ============================================================
# Page: Upload Transactions
# ============================================================
if page == "Upload Transactions":
    st.title("Upload Transactions")
    st.write("Upload a CSV file with columns: **date**, **description**, **amount**, **merchant**")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        if st.button("Upload & Process"):
            with st.spinner("Uploading..."):
                data = api("POST", "/upload", files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")})

            if data:
                st.success(data["message"])
                count = data.get("count", 0)

                # Auto-categorize
                with st.spinner("Categorizing transactions..."):
                    cat_data = api("POST", "/categorize")
                if cat_data:
                    st.success(cat_data["message"])

                # Auto-detect anomalies
                with st.spinner("Detecting anomalies..."):
                    anom_data = api("GET", "/anomalies")
                if anom_data:
                    st.success(anom_data["message"])
                    st.session_state.anomalies = anom_data.get("results", [])

                # Show uploaded transactions
                txns = data.get("transactions", [])
                if txns:
                    df = pd.DataFrame(txns)
                    # If categorize ran, refresh from the categorized results
                    if cat_data and cat_data.get("results"):
                        cat_map = {r["transaction_id"]: r["category"] for r in cat_data["results"]}
                        df["category"] = df["id"].map(cat_map).fillna(df.get("category"))
                    st.session_state.transactions = df
                    st.dataframe(df, use_container_width=True)

    # Show previously uploaded transactions
    if st.session_state.transactions is not None and uploaded_file is None:
        st.subheader("Previously Uploaded Transactions")
        st.dataframe(st.session_state.transactions, use_container_width=True)

# ============================================================
# Page: Dashboard
# ============================================================
elif page == "Dashboard":
    st.title("Spending Dashboard")

    if st.session_state.transactions is None:
        st.info("Upload transactions first to see your dashboard.")
    else:
        df = st.session_state.transactions.copy()

        # Parse amount as numeric
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        # Split into spending and income
        spending = df[df["amount"] > 0]
        income = df[df["amount"] < 0]

        total_spending = spending["amount"].sum()
        total_income = abs(income["amount"].sum())
        savings = total_income - total_spending
        savings_rate = (savings / total_income * 100) if total_income > 0 else 0

        # Summary stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Spending", f"${total_spending:,.2f}")
        col2.metric("Total Income", f"${total_income:,.2f}")
        col3.metric("Savings Rate", f"{savings_rate:.1f}%")

        # Charts
        if "category" in spending.columns and spending["category"].notna().any():
            by_cat = spending.groupby("category")["amount"].sum().reset_index()
            by_cat.columns = ["Category", "Amount"]

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig_pie = px.pie(by_cat, names="Category", values="Amount", title="Spending by Category")
                st.plotly_chart(fig_pie, use_container_width=True)

            with chart_col2:
                fig_bar = px.bar(
                    by_cat.sort_values("Amount", ascending=True),
                    x="Amount",
                    y="Category",
                    orientation="h",
                    title="Spending by Category",
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No categorized transactions yet. Upload and categorize transactions first.")

# ============================================================
# Page: Anomalies
# ============================================================
elif page == "Anomalies":
    st.title("Anomaly Detection")

    if st.button("Run Anomaly Detection"):
        with st.spinner("Detecting anomalies..."):
            data = api("GET", "/anomalies")
        if data:
            st.success(data["message"])
            st.session_state.anomalies = data.get("results", [])

    if st.session_state.anomalies:
        df_anom = pd.DataFrame(st.session_state.anomalies)
        st.subheader(f"Flagged Transactions ({len(df_anom)})")
        st.dataframe(df_anom, use_container_width=True)
    elif st.session_state.anomalies is not None:
        st.info("No anomalies detected.")
    else:
        st.info("Click 'Run Anomaly Detection' or upload transactions to see results.")

# ============================================================
# Page: AI Advice
# ============================================================
elif page == "AI Advice":
    st.title("AI Budget Advice")
    st.write("Get personalized budget recommendations powered by Google Gemini.")

    if st.button("Get Advice"):
        with st.spinner("Generating advice..."):
            data = api("POST", "/advice")
        if data:
            st.session_state.advice = data

    if st.session_state.advice:
        data = st.session_state.advice
        if data.get("cached"):
            st.caption("(cached response)")
        st.markdown(data.get("advice", ""))
