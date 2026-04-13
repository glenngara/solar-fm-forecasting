"""Ablation study: effect of fine-tuning steps on performance.

Evaluates checkpoints at 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000 steps
to show how training duration affects forecasting accuracy.
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from chronos import ChronosPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.seed import set_seed
from utils.metrics import compute_all_metrics
from utils.logger import create_log, log_step, log_metrics, log_error, save_log, save_summary, get_device
from utils.eval_utils import load_windows, get_season
from config import MODELS_DIR, TABLES_DIR, PREDICTION_LENGTHS

set_seed()

RESULTS_DIR = TABLES_DIR

# Model to run ablation on (Base gives more interesting results)
BASE_MODEL_NAME = "amazon/chronos-t5-base"
FINETUNED_DIR = MODELS_DIR / "ft-chronos-t5-base"


def evaluate_model(model_path, contexts, pred_len):
    """Run Chronos inference."""
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
    log = create_log("eval_ablation", {
        "base_model": BASE_MODEL_NAME,
        "finetuned_dir": str(FINETUNED_DIR),
        "prediction_lengths": PREDICTION_LENGTHS,
    })

    # Find all checkpoints
    checkpoints = sorted(
        FINETUNED_DIR.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1])
    )

    if not checkpoints:
        print("No checkpoints found. Run finetune_chronos.py first.")
        return

    print(f"Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  {ckpt.name}")

    # Models to evaluate: zero-shot + each checkpoint + final
    models = [("0 (zero-shot)", BASE_MODEL_NAME)]
    for ckpt in checkpoints:
        step = ckpt.name.split("-")[-1]
        models.append((step, ckpt))
    # Also add the final model if it has config.json directly
    if (FINETUNED_DIR / "config.json").exists():
        models.append(("5000 (final)", FINETUNED_DIR))

    all_results = []

    for pred_len in PREDICTION_LENGTHS:
        print(f"\n{'=' * 60}")
        print(f"Ablation — Prediction horizon: {pred_len}h")
        print(f"{'=' * 60}")

        contexts, targets, timestamps = load_windows("test", pred_len)
        timestamps = pd.DatetimeIndex(timestamps)
        seasons = np.array([get_season(m) for m in timestamps.month])

        for step_label, model_path in models:
            print(f"\n  Steps={step_label}...")
            log_step(log, f"step_{step_label}_{pred_len}h")
            try:
                preds = evaluate_model(model_path, contexts, pred_len)
            except Exception as e:
                print(f"    FAILED: {e}")
                log_error(log, f"step={step_label} {pred_len}h: {e}")
                continue

            for season_label in ["all", "dry", "wet"]:
                mask = np.ones(len(targets), dtype=bool) if season_label == "all" else (seasons == season_label)
                m = compute_all_metrics(preds[mask], targets[mask])
                all_results.append({
                    "steps": step_label,
                    "horizon": f"{pred_len}h",
                    "season": season_label,
                    **m,
                })

            overall = compute_all_metrics(preds, targets)
            log_metrics(log, f"step_{step_label}", f"{pred_len}h", "all", overall)
            print(f"    MAE: {overall['MAE']:.2f}, RMSE: {overall['RMSE']:.2f}, MASE: {overall['MASE']:.3f}")

    # Save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "ablation_steps.csv", index=False)
    save_summary(all_results, "ablation_steps")
    save_log(log)
    print(f"\nAblation results saved to results/tables/ablation_steps.csv")
    print(results_df[results_df["season"] == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
