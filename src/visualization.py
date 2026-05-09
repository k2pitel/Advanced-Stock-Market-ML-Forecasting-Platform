"""Visualization utilities for publication-ready outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns


def plot_price_history(df: pd.DataFrame, ticker: str):
    d = df[df["Ticker"] == ticker]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Close"], mode="lines", name=f"{ticker} Close"))
    fig.update_layout(template="plotly_dark", title=f"{ticker} Price History")
    return fig


def plot_candlestick(df: pd.DataFrame, ticker: str):
    d = df[df["Ticker"] == ticker]
    fig = go.Figure(data=[go.Candlestick(
        x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name=ticker
    )])
    fig.update_layout(template="plotly_dark", title=f"{ticker} Candlestick")
    return fig


def save_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="viridis")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
