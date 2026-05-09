"""Trading strategy simulation and backtesting utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    positions: pd.Series
    strategy_returns: pd.Series


def backtest_long_short(
    prices: pd.Series,
    signals: pd.Series,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> BacktestResult:
    """Backtest buy/sell/hold signals with simple risk controls.

    signals convention: 1 buy, -1 sell, 0 hold.
    """
    prices = prices.astype(float)
    signals = signals.reindex(prices.index).fillna(0).astype(int)
    raw_returns = prices.pct_change().fillna(0)

    positions = signals.replace(0, np.nan).ffill().fillna(0)
    strat_returns = positions.shift(1).fillna(0) * raw_returns

    trades = positions.diff().abs().fillna(0)
    costs = trades * (transaction_cost + slippage)
    strat_returns = strat_returns - costs

    if stop_loss is not None:
        strat_returns = strat_returns.clip(lower=-abs(stop_loss))
    if take_profit is not None:
        strat_returns = strat_returns.clip(upper=abs(take_profit))

    equity = initial_capital * (1 + strat_returns).cumprod()
    return BacktestResult(equity_curve=equity, positions=positions, strategy_returns=strat_returns)
