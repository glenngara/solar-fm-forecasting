"""Sensitivity analysis: sweep fine-tuning hyperparameters automatically.

Runs a grid of hyperparameter combinations for Chronos fine-tuning and
evaluates each on the test set. All configs are defined here — no manual tuning needed.

Hyperparameters swept:
- Learning rate: [1e-3, 1e-4, 1e-5]
- Training steps: [500, 1000, 2000]
- Context length: [72, 168, 336] (3 days, 7 days, 14 days)
"""

import sys
import itertools
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from chronos import ChronosPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.seed import set_seed
from utils.logger import create_log, log_step, log_metrics, save_log, save_summary, get_device
from utils.eval_utils import load_windows, get_season, compute_metrics
from config import PROCESSED_DIR, MODELS_DIR as _MODELS_DIR, TABLES_DIR, PREDICTION_LENGTHS

set_seed()

MODELS_DIR = _MODELS_DIR / "sensitivity"
RESULTS_DIR = TABLES_DIR
BASE_MODEL = "amazon/chronos-t5-small"

HP_GRID = {
    "learning_rate": [1e-3, 1e-4, 1e-5],
    "max_steps": [500, 1000, 2000],
    "context_length": [72, 168, 336],
}


class ChronosSweepDataset(Dataset):
    """Dataset for Chronos fine-tuning with configurable context length."""

    def __init__(self, series_list, tokenizer, context_length, prediction_length, stride=24):
        self.samples = []
        for series in series_list:
            arr = np.array(series, dtype=np.float32)
            total_len = context_length + prediction_length
            if len(arr) >= total_len:
                for start in range(0, len(arr) - total_len + 1, stride):
                    self.samples.append(arr[start : start + total_len])
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window = self.samples[idx]
        context = torch.tensor(window[: self.context_length], dtype=torch.float32)
        target = torch.tensor(
            window[self.context_length : self.context_length + self.prediction_length],
            dtype=torch.float32,
        )
        input_ids, attention_mask, tokenizer_state = (
            self.tokenizer.context_input_transform(context.unsqueeze(0))
        )
        labels, _ = self.tokenizer.label_input_transform(
            target.unsqueeze(0), tokenizer_state
        )
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }


def get_config_id(lr, steps, ctx_len):
    return f"lr{lr}_steps{steps}_ctx{ctx_len}"


def build_series_list():
    """Build training series from full training data."""
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
    print(f"Prepared {len(series_list)} training series")
    return series_list


def finetune_with_config(series_list, lr, steps, ctx_len):
    """Fine-tune Chronos with a specific hyperparameter config."""
    from transformers import Trainer, TrainingArguments

    config_id = get_config_id(lr, steps, ctx_len)
    output_path = MODELS_DIR / config_id

    if (output_path / "config.json").exists():
        print(f"  Model already exists for {config_id}, skipping training.")
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)

    pipeline = ChronosPipeline.from_pretrained(
        BASE_MODEL, device_map="cpu", dtype=torch.float32,
    )
    model = pipeline.model.model
    tokenizer = pipeline.tokenizer
    pred_len = pipeline.model.config.prediction_length

    dataset = ChronosSweepDataset(series_list, tokenizer, ctx_len, pred_len, stride=24)
    print(f"  Training: {len(dataset)} samples, lr={lr}, steps={steps}, ctx={ctx_len}")

    if len(dataset) == 0:
        print(f"  WARNING: No training samples for ctx_len={ctx_len}")
        return None

    training_args = TrainingArguments(
        output_dir=str(output_path),
        max_steps=steps,
        per_device_train_batch_size=16,
        learning_rate=lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        weight_decay=0.01,
        optim="adamw_torch",
        logging_steps=100,
        save_steps=steps,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

    trainer.save_model(str(output_path))
    model.config.save_pretrained(str(output_path))

    del pipeline, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return output_path


def evaluate_model(model_path, contexts, pred_len):
    """Run Chronos inference."""
    device = get_device()
    pipeline = ChronosPipeline.from_pretrained(
        str(model_path), device_map=device, dtype=torch.float32,
    )

    predictions = []
    for i in range(0, len(contexts), 32):
        batch = [torch.tensor(ctx, dtype=torch.float32) for ctx in contexts[i : i + 32]]
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
    })

    hp_combos = list(itertools.product(
        HP_GRID["learning_rate"],
        HP_GRID["max_steps"],
        HP_GRID["context_length"],
    ))
    print(f"Total hyperparameter combinations: {len(hp_combos)}")
    print(f"Grid: {HP_GRID}\n")

    series_list = build_series_list()
    all_results = []

    for idx, (lr, steps, ctx_len) in enumerate(hp_combos):
        config_id = get_config_id(lr, steps, ctx_len)
        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{len(hp_combos)}] Config: lr={lr}, steps={steps}, ctx_len={ctx_len}")
        print(f"{'=' * 60}")

        log_step(log, f"finetune_{config_id}", {"lr": lr, "steps": steps, "ctx_len": ctx_len})

        model_path = finetune_with_config(series_list, lr, steps, ctx_len)
        if model_path is None:
            log_step(log, f"failed_{config_id}")
            continue

        for pred_len in PREDICTION_LENGTHS:
            contexts, targets, timestamps = load_windows("test", pred_len)
            timestamps = pd.DatetimeIndex(timestamps)
            seasons = np.array([get_season(m) for m in timestamps.month])

            print(f"  Evaluating {pred_len}h horizon...")
            preds = evaluate_model(model_path, contexts, pred_len)

            for season_label in ["all", "dry", "wet"]:
                mask = np.ones(len(targets), dtype=bool) if season_label == "all" else (seasons == season_label)
                m = compute_metrics(preds[mask], targets[mask])
                all_results.append({
                    "config_id": config_id,
                    "learning_rate": lr,
                    "max_steps": steps,
                    "context_length": ctx_len,
                    "horizon": f"{pred_len}h",
                    "season": season_label,
                    **m,
                })
                log_metrics(log, config_id, f"{pred_len}h", season_label, m)

            overall = compute_metrics(preds, targets)
            print(f"    {pred_len}h MAE: {overall['MAE']:.2f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "sensitivity_results.csv", index=False)
    save_summary(all_results, "sensitivity_results")
    save_log(log)

    if not results_df.empty:
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


if __name__ == "__main__":
    main()
