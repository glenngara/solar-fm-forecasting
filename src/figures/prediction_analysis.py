"""Prediction analysis: visualize actual vs predicted for each model.

Generates diagnostic plots showing:
1. Sample 24h forecasts for all models overlaid on actuals
2. Diurnal pattern analysis (average hourly predictions)
3. Error distribution by hour of day
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, FIGURES_DIR, PROCESSED_DIR, PREDICTION_LENGTHS
from utils.eval_utils import load_windows

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Model display names mapped to prediction file patterns
MODELS_24h = {
    "Persistence": "Persistence",
    "XGBoost": "xgboost",
    "LSTM": "lstm",
    "Chronos-2": "chronos_2",
    "Chronos-T5-Small": "chronos_t5_small",
    "Chronos-T5-Base": "chronos_t5_base",
    "TimesFM-2.5": "timesfm_2.5",
    "Moirai-2.0": "moirai_2.0_small",
    "TTM-R2": "ttm_r2",
}

COLORS = {
    "Actual": "#000000",
    "Persistence": "#999999",
    "XGBoost": "#2ca02c",
    "LSTM": "#d62728",
    "Chronos-2": "#1f77b4",
    "Chronos-T5-Small": "#aec7e8",
    "Chronos-T5-Base": "#ff7f0e",
    "TimesFM-2.5": "#9467bd",
    "Moirai-2.0": "#8c564b",
    "TTM-R2": "#e377c2",
}


def load_predictions(pred_len=24):
    """Load all available prediction files."""
    preds = {}
    for name, pattern in MODELS_24h.items():
        pred_file = RESULTS_DIR / f"preds_{pattern}_{pred_len}h.npy"
        if pred_file.exists():
            preds[name] = np.load(pred_file)
        else:
            print(f"  Missing: {pred_file.name}")
    return preds


def fig_sample_forecasts(preds, targets, timestamps, pred_len=24):
    """Plot 3 sample forecast windows: dry season, wet season, and transition."""
    seasons = pd.DatetimeIndex(timestamps)

    # Pick sample indices: one dry (March), one wet (August), one transition (June)
    months = seasons.month
    dry_idx = np.where(months == 3)[0]
    wet_idx = np.where(months == 8)[0]
    trans_idx = np.where(months == 6)[0]

    samples = []
    for name, idxs in [("Dry (March)", dry_idx), ("Wet (August)", wet_idx), ("Transition (June)", trans_idx)]:
        if len(idxs) > 0:
            samples.append((name, idxs[len(idxs) // 2]))  # middle of the month

    fig, axes = plt.subplots(len(samples), 1, figsize=(14, 4 * len(samples)), sharex=False)
    if len(samples) == 1:
        axes = [axes]

    hours = np.arange(pred_len)
    for ax, (season_name, idx) in zip(axes, samples):
        actual = targets[idx]
        ax.plot(hours, actual, 'k-', linewidth=2, label="Actual", zorder=10)

        for model_name, model_preds in preds.items():
            if idx < len(model_preds):
                ax.plot(hours, model_preds[idx], '--', color=COLORS.get(model_name, '#333'),
                        linewidth=1.2, alpha=0.8, label=model_name)

        ax.set_title(f"{season_name} — {pred_len}h Forecast", fontsize=13)
        ax.set_xlabel("Forecast Hour")
        ax.set_ylabel("Solar Irradiance (W/m²)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=3, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig_sample_forecasts_{pred_len}h.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved fig_sample_forecasts_{pred_len}h.png")


def fig_diurnal_pattern(preds, targets, pred_len=24):
    """Average prediction by hour — shows if models capture the solar cycle."""
    fig, ax = plt.subplots(figsize=(12, 6))

    hours = np.arange(pred_len)
    mean_actual = targets.mean(axis=0)
    ax.plot(hours, mean_actual, 'k-', linewidth=3, label="Actual", zorder=10)

    for model_name, model_preds in preds.items():
        mean_pred = model_preds.mean(axis=0)[:pred_len]
        ax.plot(hours, mean_pred, '--', color=COLORS.get(model_name, '#333'),
                linewidth=1.5, alpha=0.8, label=f"{model_name} (MAE={np.mean(np.abs(model_preds[:,:pred_len] - targets)):.1f})")

    ax.set_xlabel("Forecast Hour", fontsize=12)
    ax.set_ylabel("Mean Solar Irradiance (W/m²)", fontsize=12)
    ax.set_title(f"Average Diurnal Forecast Pattern — {pred_len}h Horizon", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig_diurnal_pattern_{pred_len}h.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved fig_diurnal_pattern_{pred_len}h.png")


def fig_error_by_hour(preds, targets, pred_len=24):
    """MAE by forecast hour — shows where models struggle."""
    fig, ax = plt.subplots(figsize=(12, 6))

    hours = np.arange(pred_len)
    for model_name, model_preds in preds.items():
        hourly_mae = np.mean(np.abs(model_preds[:, :pred_len] - targets), axis=0)
        ax.plot(hours, hourly_mae, '-o', color=COLORS.get(model_name, '#333'),
                linewidth=1.5, markersize=3, alpha=0.8, label=model_name)

    ax.set_xlabel("Forecast Hour", fontsize=12)
    ax.set_ylabel("MAE (W/m²)", fontsize=12)
    ax.set_title(f"Error by Forecast Hour — {pred_len}h Horizon", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig_error_by_hour_{pred_len}h.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved fig_error_by_hour_{pred_len}h.png")


def main():
    print("Generating prediction analysis figures...\n")

    for pred_len in PREDICTION_LENGTHS:
        print(f"\n--- {pred_len}h horizon ---")
        contexts, targets, timestamps = load_windows("test", pred_len)
        preds = load_predictions(pred_len)

        if not preds:
            print(f"  No predictions found for {pred_len}h")
            continue

        fig_sample_forecasts(preds, targets, timestamps, pred_len)
        fig_diurnal_pattern(preds, targets, pred_len)
        fig_error_by_hour(preds, targets, pred_len)

    print("\nDone. Analysis figures saved to results/figures/")


if __name__ == "__main__":
    main()
