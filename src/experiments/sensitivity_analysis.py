"""Sensitivity analysis: sweep fine-tuning hyperparameters automatically.

Runs a grid of hyperparameter combinations for Chronos fine-tuning and
evaluates each on the test set. All configs are defined here — no manual tuning needed.

Hyperparameters swept:
- Learning rate: [1e-3, 1e-4, 1e-5]
- Training steps: [500, 1000, 2000]
- Context length: [72, 168, 336] (3 days, 7 days, 14 days)
"""

import json
import subprocess
import sys
import itertools
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from chronos import ChronosPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import create_log, log_step, log_metrics, save_log, save_summary
from utils.eval_utils import load_windows, get_season, compute_metrics
from config import PROCESSED_DIR, MODELS_DIR as _MODELS_DIR, TABLES_DIR, PREDICTION_LENGTHS

MODELS_DIR = _MODELS_DIR / "sensitivity"
RESULTS_DIR = TABLES_DIR
BASE_MODEL = "amazon/chronos-t5-small"

# ── Hyperparameter Grid ──────────────────────────────────────────────────────
# All combinations will be tried automatically.

HP_GRID = {
    "learning_rate": [1e-3, 1e-4, 1e-5],
    "max_steps": [500, 1000, 2000],
    "context_length": [72, 168, 336],
}

# Fixed parameters (not swept)
FIXED_PARAMS = {
    "batch_size": 32,
    "prediction_length": 72,
    "warmup_ratio": 0.1,
    "lr_scheduler": "linear",
}


def get_config_id(lr, steps, ctx_len):
    """Create a unique ID for this hyperparameter combo."""
    return f"lr{lr}_steps{steps}_ctx{ctx_len}"


def prepare_training_data():
    """Prepare training data if not already done."""
    data_dir = PROCESSED_DIR / "chronos_finetune"
    jsonl_path = data_dir / "train.json"

    if jsonl_path.exists():
        return jsonl_path

    data_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv", index_col="timestamp", parse_dates=True)

    series_list = []
    for year in range(2020, 2024):
        for month in range(1, 13):
            mask = (train_df.index.year == year) & (train_df.index.month == month)
            monthly = train_df.loc[mask, "ALLSKY_SFC_SW_DWN"].dropna().values
            if len(monthly) > 48:
                series_list.append(monthly.tolist())

    for year in range(2020, 2024):
        yearly = train_df.loc[train_df.index.year == year, "ALLSKY_SFC_SW_DWN"].dropna().values
        series_list.append(yearly.tolist())

    with open(jsonl_path, "w") as f:
        for i, series in enumerate(series_list):
            record = {"start": "2020-01-01 00:00:00", "target": series, "item_id": str(i)}
            f.write(json.dumps(record) + "\n")

    print(f"Prepared {len(series_list)} training series")
    return jsonl_path


def finetune_with_config(data_path, lr, steps, ctx_len):
    """Fine-tune Chronos with a specific hyperparameter config."""
    config_id = get_config_id(lr, steps, ctx_len)
    output_path = MODELS_DIR / config_id

    if output_path.exists():
        print(f"  Model already exists for {config_id}, skipping training.")
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "chronos.scripts.training",
        "--training_data_paths", str(data_path),
        "--probability", "1.0",
        "--context_length", str(ctx_len),
        "--prediction_length", str(FIXED_PARAMS["prediction_length"]),
        "--min_past", str(ctx_len),
        "--max_steps", str(steps),
        "--save_steps", str(steps),  # only save final checkpoint
        "--log_steps", "100",
        "--per_device_train_batch_size", str(FIXED_PARAMS["batch_size"]),
        "--learning_rate", str(lr),
        "--optim", "adamw_torch",
        "--shuffle_buffer_length", "10000",
        "--model_id", BASE_MODEL,
        "--model_type", "seq2seq",
        "--output_dir", str(output_path),
        "--tf32", "false",
        "--torch_compile", "false",
        "--tokenizer_class", "MeanScaleUniformBins",
        "--n_tokens", "4096",
        "--lr_scheduler_type", FIXED_PARAMS["lr_scheduler"],
        "--warmup_ratio", str(FIXED_PARAMS["warmup_ratio"]),
        "--dataloader_num_workers", "2",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:300]}")
        return None

    return output_path


def evaluate_model(model_path, contexts, pred_len):
    """Run Chronos inference."""
    from utils.logger import get_device
    device = get_device()
    pipeline = ChronosPipeline.from_pretrained(
        str(model_path), device_map=device, dtype=torch.float32,
    )

    predictions = []
    batch_size = 32
    for i in range(0, len(contexts), batch_size):
        batch = [torch.tensor(ctx, dtype=torch.float32) for ctx in contexts[i : i + batch_size]]
        forecast = pipeline.predict(batch, prediction_length=pred_len, num_samples=20)
        pred_median = forecast.median(dim=1).values.numpy()
        predictions.append(pred_median)

    del pipeline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return np.clip(np.concatenate(predictions, axis=0), 0, None)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    log = create_log("sensitivity_analysis", {
        "base_model": BASE_MODEL,
        "hp_grid": HP_GRID,
        "fixed_params": FIXED_PARAMS,
    })

    # Generate all hyperparameter combinations
    hp_combos = list(itertools.product(
        HP_GRID["learning_rate"],
        HP_GRID["max_steps"],
        HP_GRID["context_length"],
    ))
    total_combos = len(hp_combos)
    print(f"Total hyperparameter combinations: {total_combos}")
    print(f"Grid: {HP_GRID}\n")

    # Prepare training data once
    data_path = prepare_training_data()

    all_results = []

    for idx, (lr, steps, ctx_len) in enumerate(hp_combos):
        config_id = get_config_id(lr, steps, ctx_len)
        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{total_combos}] Config: lr={lr}, steps={steps}, ctx_len={ctx_len}")
        print(f"{'=' * 60}")

        log_step(log, f"finetune_{config_id}", {"lr": lr, "steps": steps, "ctx_len": ctx_len})

        # Fine-tune
        model_path = finetune_with_config(data_path, lr, steps, ctx_len)
        if model_path is None:
            log_step(log, f"failed_{config_id}")
            continue

        # Evaluate on both horizons
        for pred_len in PREDICTION_LENGTHS:
            contexts, targets, timestamps = load_windows("test", pred_len)
            timestamps = pd.DatetimeIndex(timestamps)
            seasons = np.array([get_season(m) for m in timestamps.month])

            print(f"  Evaluating {pred_len}h horizon...")
            preds = evaluate_model(model_path, contexts, pred_len)

            for season_label in ["all", "dry", "wet"]:
                mask = np.ones(len(targets), dtype=bool) if season_label == "all" else (seasons == season_label)
                m = compute_metrics(preds[mask], targets[mask])
                result = {
                    "config_id": config_id,
                    "learning_rate": lr,
                    "max_steps": steps,
                    "context_length": ctx_len,
                    "horizon": f"{pred_len}h",
                    "season": season_label,
                    **m,
                }
                all_results.append(result)
                log_metrics(log, config_id, f"{pred_len}h", season_label, m)

            overall = compute_metrics(preds, targets)
            print(f"    {pred_len}h MAE: {overall['MAE']:.2f}")

    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "sensitivity_results.csv", index=False)
    save_summary(all_results, "sensitivity_results")
    save_log(log)

    # Find best config per horizon
    print(f"\n{'=' * 60}")
    print("BEST CONFIGURATIONS")
    print(f"{'=' * 60}")
    for pred_len in PREDICTION_LENGTHS:
        for season in ["all", "wet"]:
            subset = results_df[
                (results_df["horizon"] == f"{pred_len}h") & (results_df["season"] == season)
            ]
            if subset.empty:
                continue
            best = subset.loc[subset["MAE"].idxmin()]
            print(f"\n  Best for {pred_len}h ({season}):")
            print(f"    lr={best['learning_rate']}, steps={int(best['max_steps'])}, ctx={int(best['context_length'])}")
            print(f"    MAE={best['MAE']:.2f}, RMSE={best['RMSE']:.2f}")

    print(f"\nResults saved to results/tables/sensitivity_results.csv")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
