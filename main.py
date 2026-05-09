"""Main entrypoint for data pipeline and baseline experimentation."""
from __future__ import annotations

import json
from pathlib import Path

from src.data_loader import DataLoaderConfig, download_market_data, save_raw_data
from src.evaluation import forecasting_metrics
from src.feature_engineering import engineer_features
from src.forecasting import forecast_arima
from src.preprocessing import clean_timeseries
from src.utils import configure_logging, ensure_project_dirs, set_seed


def run() -> None:
    base = Path(__file__).resolve().parent
    ensure_project_dirs(base)
    configure_logging()
    set_seed(42)

    raw_path = base / "data" / "raw" / "market_data.csv"
    processed_path = base / "data" / "processed" / "features.csv"
    metrics_path = base / "reports" / "metrics" / "baseline_forecast_metrics.json"

    data = download_market_data(DataLoaderConfig())
    save_raw_data(data, raw_path)

    clean = clean_timeseries(data)
    features = engineer_features(clean)
    features.to_csv(processed_path, index=False)

    # Simple ARIMA demo on AAPL close
    aapl = features[features["Ticker"] == "AAPL"]
    series = aapl["Close"].reset_index(drop=True)
    train = series.iloc[:-30]
    test = series.iloc[-30:]
    pred = forecast_arima(train, steps=len(test)).forecast.values
    metrics = forecasting_metrics(test.values, pred)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print("Run complete. Metrics saved to", metrics_path)


if __name__ == "__main__":
    run()
