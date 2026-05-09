"""Ensemble methods for classification and regression outputs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnsemblePrediction:
    values: np.ndarray
    method: str


def weighted_average(predictions: list[np.ndarray], weights: list[float] | None = None) -> EnsemblePrediction:
    """Weighted average ensemble for forecasting outputs."""
    preds = np.vstack(predictions)
    if weights is None:
        weights = [1 / len(predictions)] * len(predictions)
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()
    return EnsemblePrediction(np.average(preds, axis=0, weights=weights_arr), "weighted_average")


def soft_voting(probabilities: list[np.ndarray], threshold: float = 0.5) -> EnsemblePrediction:
    """Soft voting for binary classification from probability predictions."""
    mean_prob = np.mean(np.vstack(probabilities), axis=0)
    return EnsemblePrediction((mean_prob >= threshold).astype(int), "soft_voting")
