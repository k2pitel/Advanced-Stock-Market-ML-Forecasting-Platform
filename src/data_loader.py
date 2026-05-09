"""Data loading utilities for market and benchmark data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class DataLoaderConfig:
    tickers: tuple[str, ...] = ("AAPL", "TSLA", "JPM")
    benchmarks: tuple[str, ...] = ("^GSPC", "^VIX")
    period: str = "10y"
    interval: str = "1d"


def _download_single(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for symbol: {symbol}")
    df = df.rename_axis("Date").reset_index()
    df["Ticker"] = symbol
    return df


def download_market_data(config: DataLoaderConfig) -> pd.DataFrame:
    """Download OHLCV for configured stocks and benchmarks."""
    frames = [_download_single(t, config.period, config.interval) for t in (*config.tickers, *config.benchmarks)]
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return data


def save_raw_data(df: pd.DataFrame, output_path: Path) -> Path:
    """Persist raw data to csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV with date parsing fallback."""
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]) 
    return df
