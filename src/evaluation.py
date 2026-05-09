"""Evaluation metrics for classification, forecasting, and trading performance."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_prob=None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def forecasting_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    directional_accuracy = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))) if len(y_true) > 1 else 0.0
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
    }


def financial_metrics(returns: np.ndarray, risk_free_rate: float = 0.0) -> dict[str, float]:
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return {k: 0.0 for k in ["total_return", "annualized_return", "sharpe", "sortino", "volatility", "max_drawdown", "win_rate", "profit_factor", "calmar"]}

    cumulative = np.cumprod(1 + returns)
    total_return = cumulative[-1] - 1
    annualized_return = (1 + total_return) ** (252 / max(len(returns), 1)) - 1
    volatility = np.std(returns) * np.sqrt(252)
    downside = np.std(np.minimum(returns, 0)) * np.sqrt(252)
    sharpe = (annualized_return - risk_free_rate) / volatility if volatility else 0.0
    sortino = (annualized_return - risk_free_rate) / downside if downside else 0.0

    peaks = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peaks) / peaks
    max_drawdown = abs(drawdown.min())

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = float((returns > 0).mean())
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.size else float('inf')
    calmar = annualized_return / max_drawdown if max_drawdown else 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "volatility": float(volatility),
        "max_drawdown": float(max_drawdown),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "calmar": float(calmar),
    }


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
