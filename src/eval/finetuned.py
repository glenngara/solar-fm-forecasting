"""Evaluate fine-tuned Chronos models vs zero-shot vs baselines.

Compares all four Chronos variants with comprehensive metrics:
- MAE, RMSE, MASE (point metrics)
- CRPS (probabilistic calibration)
- Diebold-Mariano tests (statistical significance)
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from chronos import ChronosPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.seed import set_seed
from utils.metrics import compute_all_metrics, pairwise_dm_tests
from utils.logger import create_log, log_step, log_metrics, save_log, save_summary, get_device
from utils.eval_utils import load_windows, get_season
from config import MODELS_DIR, TABLES_DIR, PREDICTION_LENGTHS

set_seed()

RESULTS_DIR = TABLES_DIR

MODELS = [
    ("Chronos-T5-Small (zero-shot)", "amazon/chronos-t5-small"),
    ("Chronos-T5-Small (fine-tuned)", MODELS_DIR / "ft-chronos-t5-small"),
    ("Chronos-T5-Base (zero-shot)", "amazon/chronos-t5-base"),
    ("Chronos-T5-Base (fine-tuned)", MODELS_DIR / "ft-chronos-t5-base"),
]


def evaluate_model(model_path, contexts, pred_len):
    """Run Chronos inference, returning both median predictions and raw samples."""
    from utils.logger import get_device
    device = get_device()
    model_str = str(model_path)
    local_files_only = not model_str.startswith(("amazon/", "google/", "Salesforce/"))
    pipeline = ChronosPipeline.from_pretrained(
        model_str,
        device_map=device,
        dtype=torch.float32,
        local_files_only=local_files_only,
    )

    all_medians = []
    all_samples = []
    batch_size = 32
    for i in range(0, len(contexts), batch_size):
        batch = [torch.tensor(ctx, dtype=torch.float32) for ctx in contexts[i : i + batch_size]]
        forecast = pipeline.predict(batch, prediction_length=pred_len, num_samples=20)
        # forecast shape: (batch, num_samples, pred_len)
        samples_np = forecast.numpy()
        median_np = np.median(samples_np, axis=1)
        all_medians.append(median_np)
        all_samples.append(samples_np)

    medians = np.clip(np.concatenate(all_medians, axis=0), 0, None)
    samples = np.clip(np.concatenate(all_samples, axis=0), 0, None)

    del pipeline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return medians, samples


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = create_log("eval_finetuned", {
        "models": [name for name, _ in MODELS],
        "prediction_lengths": PREDICTION_LENGTHS,
    })
    all_results = []

    for pred_len in PREDICTION_LENGTHS:
        print(f"\n{'=' * 60}")
        print(f"Prediction horizon: {pred_len}h")
        print(f"{'=' * 60}")

        contexts, targets, timestamps = load_windows("test", pred_len)
        timestamps = pd.DatetimeIndex(timestamps)
        seasons = np.array([get_season(m) for m in timestamps.month])

        # Store per-window errors for DM tests
        model_errors = {}

        for model_label, model_path in MODELS:
            print(f"\nEvaluating {model_label}...")
            log_step(log, f"eval_{model_label}_{pred_len}h")
            preds, samples = evaluate_model(model_path, contexts, pred_len)

            # Save predictions and samples
            safe_name = model_label.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            np.save(RESULTS_DIR.parent / f"preds_{safe_name}_{pred_len}h.npy", preds)
            np.save(RESULTS_DIR.parent / f"samples_{safe_name}_{pred_len}h.npy", samples)

            # Store absolute errors for DM tests
            model_errors[model_label] = np.abs(preds - targets)

            # Overall and seasonal metrics
            for season_label in ["all", "dry", "wet"]:
                if season_label == "all":
                    mask = np.ones(len(targets), dtype=bool)
                else:
                    mask = seasons == season_label

                m = compute_all_metrics(
                    preds[mask], targets[mask],
                    samples=samples[mask],
                    seasonal_period=24,
                )
                all_results.append({
                    "model": model_label,
                    "horizon": f"{pred_len}h",
                    "season": season_label,
                    **m,
                })

            overall = compute_all_metrics(preds, targets, samples=samples)
            log_metrics(log, model_label, f"{pred_len}h", "all", overall)
            print(f"  MAE: {overall['MAE']:.2f}, RMSE: {overall['RMSE']:.2f}, "
                  f"MASE: {overall['MASE']:.3f}, CRPS: {overall['CRPS']:.2f}")

        # Diebold-Mariano tests
        print(f"\n{'=' * 60}")
        print(f"Diebold-Mariano Tests ({pred_len}h)")
        print(f"{'=' * 60}")
        model_names = [label for label, _ in MODELS]
        dm_results = pairwise_dm_tests(model_errors, model_names, horizon=pred_len)
        for r in dm_results:
            print(f"  {r['model_A']} vs {r['model_B']}: "
                  f"DM={r['DM_statistic']:.3f}, p={r['p_value']:.4f} {r['significance']}")

        # Save DM results
        dm_df = pd.DataFrame(dm_results)
        dm_df["horizon"] = f"{pred_len}h"
        dm_df.to_csv(RESULTS_DIR / f"dm_tests_{pred_len}h.csv", index=False)

    # Save comparison table
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "finetuned_comparison.csv", index=False)
    save_summary(all_results, "finetuned_comparison")
    save_log(log)
    print(f"\nResults saved to results/tables/finetuned_comparison.csv")
    print(results_df.to_string(index=False))

    # Improvement summary
    print(f"\n{'=' * 60}")
    print("IMPROVEMENT SUMMARY (fine-tuned vs zero-shot)")
    print(f"{'=' * 60}")
    for model_size in ["Small", "Base"]:
        print(f"\n  Chronos-T5-{model_size}:")
        for pred_len in PREDICTION_LENGTHS:
            for season in ["all", "dry", "wet"]:
                zs = results_df[
                    (results_df["model"].str.contains(f"{model_size}.*zero-shot", regex=True))
                    & (results_df["horizon"] == f"{pred_len}h")
                    & (results_df["season"] == season)
                ]
                ft = results_df[
                    (results_df["model"].str.contains(f"{model_size}.*fine-tuned", regex=True))
                    & (results_df["horizon"] == f"{pred_len}h")
                    & (results_df["season"] == season)
                ]
                for metric in ["MAE", "RMSE", "MASE", "CRPS"]:
                    zs_val = zs[metric].values[0]
                    ft_val = ft[metric].values[0]
                    imp = (1 - ft_val / zs_val) * 100
                    if metric == "MAE":
                        print(f"    {pred_len}h {season}: MAE {zs_val:.2f} → {ft_val:.2f} ({imp:+.1f}%)")


if __name__ == "__main__":
    main()
