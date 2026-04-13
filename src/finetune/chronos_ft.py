"""Fine-tune Chronos on Laguna de Bay tropical solar irradiance data.

Uses HuggingFace Trainer API directly with the Chronos tokenizer to
fine-tune both Small and Base models on local 2020-2023 training data.

Fine-tunes both models for a complete comparison:
- Small zero-shot vs Small fine-tuned (effect of fine-tuning)
- Base zero-shot vs Base fine-tuned (effect of fine-tuning)
- Small fine-tuned vs Base fine-tuned (effect of model size)
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.seed import set_seed
from utils.logger import create_log, log_step, save_log
from config import PROCESSED_DIR, MODELS_DIR, FINETUNE_CHRONOS

set_seed()

OUTPUT_DIR = MODELS_DIR
MODELS_TO_FINETUNE = FINETUNE_CHRONOS

MAX_STEPS = 5000


class ChronosFineTuneDataset(Dataset):
    """Dataset that yields (input_ids, attention_mask, labels) for Chronos fine-tuning."""

    def __init__(self, series_list, tokenizer, context_length, prediction_length,
                 stride=None):
        self.samples = []
        if stride is None:
            stride = prediction_length
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
        target = torch.tensor(window[self.context_length :], dtype=torch.float32)

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


def build_series_list():
    """Load training data and split into multiple time series."""
    train_df = pd.read_csv(
        PROCESSED_DIR / "train.csv", index_col="timestamp", parse_dates=True
    )

    series_list = []
    # Monthly chunks for seasonal diversity
    for year in range(2020, 2024):
        for month in range(1, 13):
            mask = (train_df.index.year == year) & (train_df.index.month == month)
            monthly = train_df.loc[mask, "ALLSKY_SFC_SW_DWN"].dropna().values
            if len(monthly) > 48:
                series_list.append(monthly.tolist())

    # Full-year series for long-range patterns
    for year in range(2020, 2024):
        yearly = train_df.loc[
            train_df.index.year == year, "ALLSKY_SFC_SW_DWN"
        ].dropna().values
        series_list.append(yearly.tolist())

    print(f"Created {len(series_list)} training series")
    return series_list


def finetune_model(base_model, output_name, series_list):
    """Fine-tune a single Chronos model."""
    from chronos import ChronosPipeline
    from transformers import Trainer, TrainingArguments

    output_path = OUTPUT_DIR / output_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Fine-tuning {base_model} → {output_name}")
    print(f"Steps: {MAX_STEPS}")
    print(f"{'=' * 60}")

    pipeline = ChronosPipeline.from_pretrained(
        base_model,
        device_map="cpu",
        dtype=torch.float32,
    )
    model = pipeline.model.model
    tokenizer = pipeline.tokenizer

    model_pred_len = pipeline.model.config.prediction_length
    model_ctx_len = pipeline.model.config.context_length
    print(f"Model config: context_length={model_ctx_len}, prediction_length={model_pred_len}")

    dataset = ChronosFineTuneDataset(
        series_list, tokenizer, model_ctx_len, model_pred_len,
        stride=24,
    )
    print(f"Training samples (windows): {len(dataset)}")

    if len(dataset) == 0:
        print("ERROR: No training samples created.")
        sys.exit(1)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=16,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="adamw_torch",
        logging_steps=100,
        save_steps=500,
        save_total_limit=10,
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"Saving model to {output_path}...")
    trainer.save_model(str(output_path))
    model.config.save_pretrained(str(output_path))

    if not (output_path / "config.json").exists():
        print(f"ERROR: No config.json found in {output_path}")
        sys.exit(1)

    print(f"Fine-tuning complete! Model saved to {output_path}")


def main():
    log = create_log("finetune_chronos", {
        "models": [m for m, _ in MODELS_TO_FINETUNE],
        "max_steps": MAX_STEPS,
    })
    series_list = build_series_list()
    log_step(log, "data_prepared", {"num_series": len(series_list)})

    for base_model, output_name in MODELS_TO_FINETUNE:
        log_step(log, f"finetune_start_{output_name}", {"base_model": base_model})
        finetune_model(base_model, output_name, series_list)
        log_step(log, f"finetune_done_{output_name}")

    save_log(log)
    print(f"\nAll fine-tuning complete!")
    print(f"Models saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
