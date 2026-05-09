"""Model explainability helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_scores(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Compute permutation importances in a model-agnostic way."""
    res = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)
    return pd.DataFrame({"feature": X.columns, "importance": res.importances_mean}).sort_values("importance", ascending=False)


def shap_values(model, X: pd.DataFrame):
    """Return SHAP values when shap is available."""
    try:
        import shap
    except Exception:
        return None
    explainer = shap.Explainer(model, X)
    return explainer(X)
