"""Forecasting wrappers for statistical and ML models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class ForecastResult:
    model_name: str
    forecast: pd.Series


def forecast_arima(series: pd.Series, order: tuple[int, int, int] = (5, 1, 0), steps: int = 30) -> ForecastResult:
    """ARIMA-based direct horizon forecast."""
    model = ARIMA(series, order=order)
    fit = model.fit()
    pred = fit.forecast(steps=steps)
    return ForecastResult("ARIMA", pd.Series(pred))


def forecast_exp_smoothing(series: pd.Series, steps: int = 30) -> ForecastResult:
    """Exponential smoothing forecast."""
    fit = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
    pred = fit.forecast(steps)
    return ForecastResult("ExponentialSmoothing", pd.Series(pred))


def recursive_forecast(last_window: np.ndarray, model, horizon: int = 30) -> np.ndarray:
    """Recursive forecasting for sklearn-like regressors."""
    window = last_window.astype(float).copy()
    preds = []
    for _ in range(horizon):
        next_val = float(model.predict(window.reshape(1, -1))[0])
        preds.append(next_val)
        window = np.roll(window, -1)
        window[-1] = next_val
    return np.asarray(preds)


def forecast_prophet(df: pd.DataFrame, horizon: int = 30) -> Optional[pd.DataFrame]:
    """Prophet forecast if Prophet is installed."""
    try:
        from prophet import Prophet
    except Exception:
        return None

    p_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(p_df)
    future = model.make_future_dataframe(periods=horizon)
    return model.predict(future)
