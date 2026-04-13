"""Shared utilities for evaluation scripts.

Provides common functions used across all eval and fine-tuning evaluation scripts.
"""

import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def load_windows(split, pred_len):
    """Load pre-computed forecast windows (contexts, targets, timestamps)."""
    data = np.load(PROCESSED_DIR / f"{split}_windows_{pred_len}h.npz", allow_pickle=True)
    return data["contexts"], data["targets"], data["timestamps"]


def get_season(month):
    """Philippine seasons: dry (Dec-May), wet/monsoon (Jun-Nov)."""
    return "dry" if month in [12, 1, 2, 3, 4, 5] else "wet"


def compute_metrics(predictions, targets, samples=None):
    """Compute all metrics (MAE, RMSE, MASE, optionally CRPS)."""
    from utils.metrics import compute_all_metrics
    return compute_all_metrics(predictions, targets, samples=samples, seasonal_period=24)


def persistence_forecast(contexts, pred_len):
    """Naive baseline: repeat the last pred_len hours of context."""
    predictions = []
    for ctx in contexts:
        pred = np.tile(ctx[-pred_len:], (pred_len // len(ctx[-pred_len:]) + 1))[:pred_len]
        predictions.append(pred)
    return np.array(predictions)
