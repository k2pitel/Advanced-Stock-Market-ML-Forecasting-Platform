# Advanced Machine Learning and Deep Learning for Stock Market Forecasting and Trading Strategy Optimization

A research-grade, modular Python platform for stock market forecasting, directional prediction, ensemble modeling, explainable AI, and trading strategy simulation across **AAPL**, **TSLA**, and **JPM**, with **VIX** and **S&P 500** context.

## Features

- Automated market data collection with `yfinance`
- Robust preprocessing (missing data, duplicates, chronology, outlier clipping, scaling)
- Technical indicators (RSI, MACD, Bollinger Bands, SMA/EMA, ROC, ATR, OBV, Stochastic, Williams %R)
- Feature engineering (returns, lags, rolling statistics, trend/interaction features)
- Classification and forecasting support
- Ensemble utilities (weighted averaging, soft voting)
- Trading simulation with transaction costs, slippage, and optional stop-loss/take-profit
- Financial and ML evaluation metrics
- Explainability hooks (permutation importance + SHAP when available)
- Streamlit dashboard scaffold
- Notebook workflow for end-to-end experimentation
- Poster layout asset for academic presentation

## Project Structure

```text
project_root/
├── data/{raw,processed,external}
├── notebooks/
├── src/
├── models/{trained_models,scalers}
├── reports/{figures,metrics,poster}
├── dashboard/streamlit_app.py
├── tests/
├── requirements.txt
├── README.md
└── main.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

1. Run data + baseline pipeline:

```bash
python main.py
```

2. Launch dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

3. Run tests:

```bash
python -m unittest discover -s tests -q
```

## Modeling Scope

### Classification
- Binary next-day direction (`target_up_next_day`)
- Optional multiclass direction (`target_multiclass`)

### Forecasting
- Next-day and multi-step forecasting (up to 30-day horizon)
- Recursive and direct-style helper support
- ARIMA, Exponential Smoothing, Prophet (optional dependency availability)

### Deep Learning
- MLP, LSTM, GRU model builders in `src/deep_learning.py`
- Sequence generation utility for sliding windows

## Evaluation

- Classification: Accuracy, Precision, Recall, F1, ROC-AUC
- Forecasting: RMSE, MAE, MAPE, R², directional accuracy
- Finance: total return, annualized return, Sharpe, Sortino, volatility, max drawdown, win rate, profit factor, Calmar

## Time-Series Validation Principles

- Chronological split only (no random shuffle)
- Leakage-aware scaling
- Lag/rolling features generated from historical windows

## Research Discussion

This project is designed around known financial ML constraints:
- Efficient market hypothesis implications
- Non-stationarity and regime shifts
- Volatility clustering
- Data leakage and look-ahead bias risk
- Overfitting risk under noisy, non-iid data
- Ethical use and limitations of predictive systems in finance

## Poster and Presentation Support

- Poster structure template: `reports/poster/poster_layout.md`
- Notebook narrative flow: `notebooks/01` → `notebooks/09`

## Future Improvements

- Reinforcement learning trading agents
- Transformer/attention forecasting
- News sentiment and alternative data integration
- Portfolio optimization and risk parity
- Real-time streaming inference architecture

## License

This repository is provided for academic and portfolio use. Add a formal license file as needed.
