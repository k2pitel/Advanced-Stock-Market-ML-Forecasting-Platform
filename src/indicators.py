"""Technical indicator feature generation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def add_indicators(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Add core technical indicators required by the project brief."""
    out = df.copy()
    close = out[price_col]
    high = out.get("High", close)
    low = out.get("Low", close)
    vol = out.get("Volume", pd.Series(0, index=out.index))

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    loss_safe = loss.replace(0, 1e-12)
    rs = gain / loss_safe
    out["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    out["MACD"] = ema12 - ema26
    out["MACD_signal"] = _ema(out["MACD"], 9)
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    out["BB_mid"] = mid
    out["BB_upper"] = mid + 2 * std
    out["BB_lower"] = mid - 2 * std

    for w in (5, 10, 20, 50, 100, 200):
        out[f"SMA_{w}"] = close.rolling(w).mean()
        out[f"EMA_{w}"] = _ema(close, w)
        out[f"ROC_{w}"] = close.pct_change(w)
        out[f"VOL_{w}"] = close.pct_change().rolling(w).std() * np.sqrt(252)

    out["Momentum_10"] = close - close.shift(10)

    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    out["ATR_14"] = tr.rolling(14).mean()

    direction = np.sign(close.diff()).fillna(0)
    out["OBV"] = (direction * vol).cumsum()

    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    out["STOCH_K"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    out["STOCH_D"] = out["STOCH_K"].rolling(3).mean()
    out["WilliamsR_14"] = -100 * (high14 - close) / (high14 - low14).replace(0, np.nan)
    return out
