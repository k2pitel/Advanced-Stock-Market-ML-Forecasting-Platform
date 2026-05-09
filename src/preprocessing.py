"""Time-series preprocessing and validation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def clean_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Apply date parsing, deduping, sorting, and missing-value handling."""
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"])  # robust date formatting
    out = out.drop_duplicates().sort_values([c for c in ["Ticker", "Date"] if c in out.columns])
    out = out.reset_index(drop=True)
    out = out.ffill()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].interpolate(method="linear", limit_direction="forward")
    return out


def clip_outliers(df: pd.DataFrame, cols: Iterable[str], z: float = 3.5) -> pd.DataFrame:
    """Clip extreme values using robust z-score style IQR rule."""
    out = df.copy()
    for col in cols:
        q1, q3 = out[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - z * iqr, q3 + z * iqr
        out[col] = out[col].clip(lower=lower, upper=upper)
    return out


def add_lagged_features(df: pd.DataFrame, cols: Iterable[str], lags: Iterable[int] = (1, 5, 21)) -> pd.DataFrame:
    """Create lagged feature columns (1-day, 1-week, 1-month trading days)."""
    out = df.copy()
    grouped = out.groupby("Ticker") if "Ticker" in out.columns else [(None, out)]
    if isinstance(grouped, list):
        for col in cols:
            for lag in lags:
                out[f"{col}_lag_{lag}"] = out[col].shift(lag)
        return out
    for col in cols:
        for lag in lags:
            out[f"{col}_lag_{lag}"] = grouped[col].shift(lag)
    return out


def add_rolling_stats(df: pd.DataFrame, col: str = "Close", windows: Iterable[int] = (5, 10, 20, 50, 100, 200)) -> pd.DataFrame:
    """Add rolling mean and std features for configured windows."""
    out = df.copy()
    grouped = out.groupby("Ticker") if "Ticker" in out.columns else [(None, out)]
    if isinstance(grouped, list):
        for w in windows:
            out[f"{col}_roll_mean_{w}"] = out[col].rolling(w).mean()
            out[f"{col}_roll_std_{w}"] = out[col].rolling(w).std()
        return out
    for w in windows:
        out[f"{col}_roll_mean_{w}"] = grouped[col].transform(lambda s: s.rolling(w).mean())
        out[f"{col}_roll_std_{w}"] = grouped[col].transform(lambda s: s.rolling(w).std())
    return out


def time_aware_split(df: pd.DataFrame, target: str, test_size: float = 0.2) -> SplitData:
    """Chronological split without shuffling."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    n = len(df)
    split_idx = int(n * (1 - test_size))
    X = df.drop(columns=[target])
    y = df[target]
    return SplitData(X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:])


def scale_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit scaler on train and transform train/test to avoid leakage."""
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns, index=train.index)
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)
    return train_scaled, test_scaled, scaler
