"""Data efficiency experiment: how many months of local data are needed for fine-tuning?

Fine-tunes Chronos with increasing amounts of training data:
- 3 months, 6 months, 12 months, 24 months, 48 months (full)
Evaluates each on the same test set to show the learning curve.
"""

import json
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from chronos import ChronosPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import create_log, log_step, log_metrics, save_log, save_summary
from utils.eval_utils import load_windows, get_season, compute_metrics
from config import PROCESSED_DIR, MODELS_DIR, TABLES_DIR, PREDICTION_LENGTHS

RESULTS_DIR = TABLES_DIR
BASE_MODEL = "amazon/chronos-t5-small"

# Training data subsets (months from start of 2020)
DATA_SUBSETS = [3, 6, 12, 24, 48]


def create_subset_data(n_months):
    """Create a training data file using only the first n_months of data."""
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv", index_col="timestamp", parse_dates=True)

    # Subset to first n_months
    start = train_df.index.min()
    end = start + pd.DateOffset(months=n_months)
    subset = train_df.loc[:end]

    # Create series (monthly chunks)
    series_list = []
    for year in subset.index.year.unique():
        for month in subset.index.month.unique():
            mask = (subset.index.year == year) & (subset.index.month == month)
            monthly = subset.loc[mask, "ALLSKY_SFC_SW_DWN"].dropna().values
            if len(monthly) > 48:
                series_list.append(monthly.tolist())

    # Save as JSON lines
    data_dir = PROCESSED_DIR / "chronos_finetune"
    data_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = data_dir / f"train_{n_months}m.json"
    with open(jsonl_path, "w") as f:
        for i, series in enumerate(series_list):
            record = {"start": "2020-01-01 00:00:00", "target": series, "item_id": str(i)}
            f.write(json.dumps(record) + "\n")

    print(f"  {n_months} months: {len(series_list)} series, {sum(len(s) for s in series_list)} total hours")
    return jsonl_path


def finetune_subset(n_months, data_path):
    """Fine-tune Chronos on a data subset."""
    output_path = MODELS_DIR / f"chronos-t5-small-{n_months}m"

    # Scale steps with data size
    max_steps = min(500 + n_months * 30, 2000)

    cmd = [
        sys.executable, "-m", "chronos.scripts.training",
        "--training_data_paths", str(data_path),
        "--probability", "1.0",
        "--context_length", "168",
        "--prediction_length", "72",
        "--min_past", "168",
        "--max_steps", str(max_steps),
        "--save_steps", str(max_steps),  # only save final
        "--log_steps", "100",
        "--per_device_train_batch_size", "32",
        "--learning_rate", "1e-4",
        "--optim", "adamw_torch",
        "--shuffle_buffer_length", "5000",
        "--model_id", BASE_MODEL,
        "--model_type", "seq2seq",
        "--output_dir", str(output_path),
        "--tf32", "false",
        "--torch_compile", "false",
        "--tokenizer_class", "MeanScaleUniformBins",
        "--n_tokens", "4096",
        "--lr_scheduler_type", "linear",
        "--warmup_ratio", "0.1",
    ]

    print(f"  Fine-tuning with {n_months} months ({max_steps} steps)...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  WARNING: Fine-tuning failed for {n_months}m: {result.stderr[:200]}")
        return None

    return output_path


def evaluate_model(model_path, contexts, pred_len):
    """Run inference."""
    from logger import get_device
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

    predictions = np.concatenate(predictions, axis=0)
    return np.clip(predictions, 0, None)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log = create_log("data_efficiency", {
        "data_subsets_months": DATA_SUBSETS,
        "prediction_lengths": PREDICTION_LENGTHS,
        "base_model": BASE_MODEL,
    })
    all_results = []

    # Prepare all data subsets
    print("Preparing data subsets...")
    data_paths = {}
    for n_months in DATA_SUBSETS:
        data_paths[n_months] = create_subset_data(n_months)

    # Fine-tune and evaluate each subset
    for n_months in DATA_SUBSETS:
        print(f"\n{'=' * 40}")
        print(f"Data subset: {n_months} months")
        print(f"{'=' * 40}")

        log_step(log, f"finetune_{n_months}m")
        model_path = finetune_subset(n_months, data_paths[n_months])
        if model_path is None:
            continue

        for pred_len in PREDICTION_LENGTHS:
            contexts, targets, timestamps = load_windows("test", pred_len)
            timestamps = pd.DatetimeIndex(timestamps)
            seasons = np.array([get_season(m) for m in timestamps.month])

            preds = evaluate_model(model_path, contexts, pred_len)

            for season_label in ["all", "dry", "wet"]:
                mask = np.ones(len(targets), dtype=bool) if season_label == "all" else (seasons == season_label)
                m = compute_metrics(preds[mask], targets[mask])
                all_results.append({
                    "months_training": n_months,
                    "horizon": f"{pred_len}h",
                    "season": season_label,
                    **m,
                })

            overall = compute_metrics(preds, targets)
            log_metrics(log, f"{n_months}m", f"{pred_len}h", "all", overall)
            print(f"  {pred_len}h MAE: {overall['MAE']:.2f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "data_efficiency_results.csv", index=False)
    save_summary(all_results, "data_efficiency_results")
    save_log(log)
    print(f"\nResults saved to results/tables/data_efficiency_results.csv")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
