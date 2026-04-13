"""Prepare NASA POWER data for forecasting experiments.

Splits data into train/val/test and creates forecasting windows.
- Train: 2020-2023 (for fine-tuning)
- Validation: 2024
- Test: 2025
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, PROCESSED_DIR, CONTEXT_LENGTH, PREDICTION_LENGTHS, TARGET_COL, RAW_FILENAME

RAW_PATH = RAW_DIR / RAW_FILENAME


def load_and_clean():
    """Load raw CSV and handle missing values."""
    df = pd.read_csv(RAW_PATH, index_col="timestamp", parse_dates=True)
    # Interpolate small gaps (max 3 consecutive hours)
    df[TARGET_COL] = df[TARGET_COL].interpolate(method="linear", limit=3)
    # Drop any remaining NaN rows
    df = df.dropna(subset=[TARGET_COL])
    return df


def split_data(df):
    """Split into train/val/test by year."""
    train = df[df.index.year <= 2023]
    val = df[df.index.year == 2024]
    test = df[df.index.year == 2025]
    return train, val, test


def create_forecast_windows(df, context_len, prediction_len, stride=24):
    """Create sliding window forecasting samples.

    Returns:
        contexts: list of np.array, each shape (context_len,)
        targets: list of np.array, each shape (prediction_len,)
        timestamps: list of pd.Timestamp (start of prediction window)
    """
    values = df[TARGET_COL].values
    timestamps = df.index

    contexts = []
    targets = []
    pred_timestamps = []

    for i in range(0, len(values) - context_len - prediction_len + 1, stride):
        ctx = values[i : i + context_len]
        tgt = values[i + context_len : i + context_len + prediction_len]
        ts = timestamps[i + context_len]

        # Skip windows with NaN
        if np.isnan(ctx).any() or np.isnan(tgt).any():
            continue

        contexts.append(ctx)
        targets.append(tgt)
        pred_timestamps.append(ts)

    return contexts, targets, pred_timestamps


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_and_clean()
    print(f"Total records after cleaning: {len(df)}")

    train, val, test = split_data(df)
    print(f"Train: {len(train)} ({train.index.min()} to {train.index.max()})")
    print(f"Val:   {len(val)} ({val.index.min()} to {val.index.max()})")
    print(f"Test:  {len(test)} ({test.index.min()} to {test.index.max()})")

    # Save splits as CSV
    train.to_csv(PROCESSED_DIR / "train.csv")
    val.to_csv(PROCESSED_DIR / "val.csv")
    test.to_csv(PROCESSED_DIR / "test.csv")

    # Create and save forecast windows for each prediction length
    for pred_len in PREDICTION_LENGTHS:
        for split_name, split_df in [("val", val), ("test", test)]:
            contexts, targets, timestamps = create_forecast_windows(
                split_df, CONTEXT_LENGTH, pred_len
            )
            np.savez(
                PROCESSED_DIR / f"{split_name}_windows_{pred_len}h.npz",
                contexts=np.array(contexts),
                targets=np.array(targets),
                timestamps=np.array(timestamps, dtype="datetime64[ns]"),
            )
            print(f"{split_name} {pred_len}h: {len(contexts)} windows")

    # Save full training series for fine-tuning (single continuous array)
    train_series = train[TARGET_COL].values
    np.save(PROCESSED_DIR / "train_series.npy", train_series)
    print(f"\nTrain series length: {len(train_series)} hours")

    # Seasonal stats for analysis
    print("\n--- Seasonal Irradiance Stats (Test 2025) ---")
    test_monthly = test.groupby(test.index.month)[TARGET_COL].agg(["mean", "std"])
    test_monthly.index.name = "month"
    print(test_monthly.round(2))


if __name__ == "__main__":
    main()
