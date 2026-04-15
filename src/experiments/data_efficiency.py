"""Data efficiency experiment: how many months of local data are needed for fine-tuning?

Fine-tunes Chronos with increasing amounts of training data:
- 3 months, 6 months, 12 months, 24 months, 48 months (full)
Evaluates each on the same test set to show the learning curve.
"""

import sys
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
from config import PROCESSED_DIR, MODELS_DIR, TABLES_DIR, PREDICTION_LENGTHS

set_seed()

RESULTS_DIR = TABLES_DIR
BASE_MODEL = "amazon/chronos-t5-small"
DATA_SUBSETS = [3, 6, 12, 24, 48]


class ChronosSubsetDataset(Dataset):
    """Dataset for Chronos fine-tuning from a data subset."""

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


def build_subset_series(n_months):
    """Build training series from first n_months of data."""
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv", index_col="timestamp", parse_dates=True)
    start = train_df.index.min()
    end = start + pd.DateOffset(months=n_months)
    subset = train_df.loc[:end]

    series_list = []
    for year in subset.index.year.unique():
        for month in range(1, 13):
            mask = (subset.index.year == year) & (subset.index.month == month)
            monthly = subset.loc[mask, "ALLSKY_SFC_SW_DWN"].dropna().values
            if len(monthly) > 48:
                series_list.append(monthly.tolist())

    print(f"  {n_months} months: {len(series_list)} series, {sum(len(s) for s in series_list)} total hours")
    return series_list


def finetune_subset(n_months, series_list):
    """Fine-tune Chronos on a data subset using HuggingFace Trainer."""
    from transformers import Trainer, TrainingArguments

    output_path = MODELS_DIR / f"chronos-t5-small-{n_months}m"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    max_steps = min(500 + n_months * 30, 2000)

    pipeline = ChronosPipeline.from_pretrained(
        BASE_MODEL, device_map="cpu", dtype=torch.float32,
    )
    model = pipeline.model.model
    tokenizer = pipeline.tokenizer
    ctx_len = pipeline.model.config.context_length
    pred_len = pipeline.model.config.prediction_length

    dataset = ChronosSubsetDataset(series_list, tokenizer, ctx_len, pred_len, stride=24)
    print(f"  Fine-tuning with {n_months} months ({max_steps} steps, {len(dataset)} samples)...")

    if len(dataset) == 0:
        print(f"  WARNING: No training samples for {n_months}m")
        return None

    training_args = TrainingArguments(
        output_dir=str(output_path),
        max_steps=max_steps,
        per_device_train_batch_size=16,
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        weight_decay=0.01,
        optim="adamw_torch",
        logging_steps=100,
        save_steps=max_steps,
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
    print(f"  Model saved to {output_path}")

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
    log = create_log("data_efficiency", {
        "data_subsets_months": DATA_SUBSETS,
        "prediction_lengths": PREDICTION_LENGTHS,
        "base_model": BASE_MODEL,
    })
    all_results = []

    # Build all data subsets
    print("Preparing data subsets...")
    subset_series = {}
    for n_months in DATA_SUBSETS:
        subset_series[n_months] = build_subset_series(n_months)

    # Fine-tune and evaluate each subset
    for n_months in DATA_SUBSETS:
        print(f"\n{'=' * 40}")
        print(f"Data subset: {n_months} months")
        print(f"{'=' * 40}")

        log_step(log, f"finetune_{n_months}m")
        model_path = finetune_subset(n_months, subset_series[n_months])
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
