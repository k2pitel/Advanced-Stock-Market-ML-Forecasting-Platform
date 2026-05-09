"""Traditional ML model builders and evaluation pipeline."""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ModelSpec:
    name: str
    estimator: object


def baseline_model_specs() -> list[ModelSpec]:
    """Return required baseline model configurations."""
    return [
        ModelSpec("logistic_regression", Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])),
        ModelSpec("random_forest", RandomForestClassifier(n_estimators=200, random_state=42)),
        ModelSpec("svm", Pipeline([("scaler", StandardScaler()), ("model", SVC(probability=True, kernel="rbf", C=1.0))])),
        ModelSpec("decision_tree", DecisionTreeClassifier(random_state=42)),
        ModelSpec("gradient_boosting", GradientBoostingClassifier(random_state=42)),
        ModelSpec("knn", Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=7))])),
        ModelSpec("naive_bayes", GaussianNB()),
    ]


def optional_boosting_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    try:
        from xgboost import XGBClassifier
        specs.append(ModelSpec("xgboost", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42)))
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        specs.append(ModelSpec("lightgbm", LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)))
    except Exception:
        pass
    return specs
