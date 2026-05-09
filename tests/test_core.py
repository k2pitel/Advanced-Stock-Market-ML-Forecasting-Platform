import unittest

import numpy as np
import pandas as pd

from src.evaluation import financial_metrics
from src.feature_engineering import engineer_features
from src.preprocessing import clean_timeseries, time_aware_split


class CorePipelineTests(unittest.TestCase):
    def test_clean_timeseries_sorts_and_parses_dates(self):
        df = pd.DataFrame({
            "Date": ["2024-01-02", "2024-01-01"],
            "Ticker": ["AAPL", "AAPL"],
            "Close": [101.0, 100.0],
            "Open": [100.0, 99.0],
            "High": [102.0, 101.0],
            "Low": [99.0, 98.0],
            "Adj Close": [101.0, 100.0],
            "Volume": [1000, 1000],
        })
        out = clean_timeseries(df)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(out["Date"]))
        self.assertLessEqual(out["Date"].iloc[0], out["Date"].iloc[-1])

    def test_feature_engineering_creates_targets(self):
        n = 260
        base = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "Ticker": ["AAPL"] * n,
            "Open": np.linspace(100, 150, n),
            "High": np.linspace(101, 151, n),
            "Low": np.linspace(99, 149, n),
            "Close": np.linspace(100, 150, n),
            "Adj Close": np.linspace(100, 150, n),
            "Volume": np.linspace(1000, 2000, n),
        })
        feats = engineer_features(base)
        self.assertIn("target_up_next_day", feats.columns)
        self.assertIn("RSI_14", feats.columns)
        self.assertGreater(len(feats), 0)

    def test_time_aware_split_keeps_order(self):
        df = pd.DataFrame({"x": range(10), "target": range(10)})
        split = time_aware_split(df, "target", test_size=0.2)
        self.assertEqual(len(split.X_train), 8)
        self.assertEqual(len(split.X_test), 2)
        self.assertEqual(split.X_train.index.max(), 7)
        self.assertEqual(split.X_test.index.min(), 8)

    def test_financial_metrics_runs(self):
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        metrics = financial_metrics(returns)
        self.assertIn("sharpe", metrics)
        self.assertIn("max_drawdown", metrics)


if __name__ == "__main__":
    unittest.main()
