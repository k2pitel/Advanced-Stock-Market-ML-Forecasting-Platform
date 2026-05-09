"""Feature engineering for returns, volatility, trends, and targets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import add_indicators
from .preprocessing import add_lagged_features, add_rolling_stats


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build project-level financial feature set without leakage."""
    out = df.copy()
    out = add_indicators(out)

    out["daily_return"] = out["Close"].pct_change()
    out["log_return"] = np.log(out["Close"]).diff()
    out["rolling_return_20"] = out["Close"].pct_change(20)
    out["cumulative_return"] = (1 + out["daily_return"].fillna(0)).cumprod() - 1

    out = add_rolling_stats(out, "Close")
    out = add_lagged_features(out, ["Close", "Volume"], lags=(1, 5, 21))

    out["volatility_cluster_20"] = out["daily_return"].rolling(20).std()
    out["trend_strength_20"] = (out["SMA_20"] - out["SMA_50"]) / out["SMA_50"].replace(0, np.nan)
    out["rsi_macd_interaction"] = out["RSI_14"] * out["MACD"]

    out["target_up_next_day"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    change = out["Close"].pct_change().shift(-1)
    bins = [-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf]
    labels = [0, 1, 2, 3, 4]
    out["target_multiclass"] = pd.cut(change, bins=bins, labels=labels).astype("float")

    return out.dropna().reset_index(drop=True)
