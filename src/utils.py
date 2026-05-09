"""Utility helpers for reproducibility, paths, and logging."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np


def configure_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for project scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_project_dirs(base: Path) -> None:
    """Create standard project directories if they do not exist."""
    for rel in [
        "data/raw",
        "data/processed",
        "data/external",
        "models/trained_models",
        "models/scalers",
        "reports/figures",
        "reports/metrics",
        "reports/poster",
    ]:
        (base / rel).mkdir(parents=True, exist_ok=True)
