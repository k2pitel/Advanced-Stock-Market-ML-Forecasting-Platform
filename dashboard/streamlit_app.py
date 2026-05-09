"""Streamlit dashboard for interactive stock analysis and forecasting."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.visualization import plot_candlestick, plot_price_history


st.set_page_config(page_title="Stock ML Forecasting Dashboard", layout="wide")
st.title("Advanced Stock Market ML Forecasting Dashboard")

base = Path(__file__).resolve().parents[1]
data_path = base / "data" / "processed" / "features.csv"

if not data_path.exists():
    st.warning("Processed data not found. Run main.py first.")
    st.stop()


df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"])

stock = st.sidebar.selectbox("Stock", sorted(df["Ticker"].unique()))

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_price_history(df, stock), use_container_width=True)
with col2:
    st.plotly_chart(plot_candlestick(df, stock), use_container_width=True)

st.subheader("Latest Metrics")
latest = df[df["Ticker"] == stock].tail(1).T
st.dataframe(latest)
