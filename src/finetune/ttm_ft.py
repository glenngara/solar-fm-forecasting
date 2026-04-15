"""Fine-tune TTM-R2 (IBM Granite) on Laguna de Bay tropical solar irradiance data.

Uses standard HuggingFace Trainer API. TTM-R2 is an MLP-Mixer architecture
(~1M params) that supports efficient fine-tuning on small datasets.
"""

import sys
import numpy as np
import pandas as pd
import torch
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.seed import set_seed
from utils.logger import create_log, log_step, save_log
from config import PROCESSED_DIR, MODELS_DIR, FINETUNE_TTM, CONTEXT_LENGTH, PREDICTION_LENGTHS

set_seed()

MODEL_ID, FINETUNED_NAME = FINETUNE_TTM
OUTPUT_DIR = MODELS_DIR
MAX_EPOCHS = 50


def build_datasets(pred_len):
    """Build train/test datasets for TTM fine-tuning."""
    train_df = pd.read_csv(
        PROCESSED_DIR / "train.csv", index_col="timestamp", parse_dates=True
    )
    values = train_df["ALLSKY_SFC_SW_DWN"].interpolate(limit=3).dropna().values

    # Create sliding windows
    total_len = CONTEXT_LENGTH + pred_len
    contexts, targets = [], []
    for start in range(0, len(values) - total_len + 1, 24):
        ctx = values[start : start + CONTEXT_LENGTH]
        tgt = values[start + CONTEXT_LENGTH : start + total_len]
        if not (np.isnan(ctx).any() or np.isnan(tgt).any()):
            contexts.append(ctx)
            targets.append(tgt)

    return np.array(contexts, dtype=np.float32), np.array(targets, dtype=np.float32)


def finetune():
    """Fine-tune TTM-R2 using HuggingFace Trainer."""
    from tsfm_public.toolkit.get_model import get_model
    from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
    from torch.utils.data import Dataset

    output_path = OUTPUT_DIR / FINETUNED_NAME
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fine-tuning {MODEL_ID}...")
    print(f"Output: {output_path}")

    class SolarDataset(Dataset):
        def __init__(self, contexts, targets, ctx_len):
            # Pad contexts to ctx_len if shorter
            padded = []
            for ctx in contexts:
                if len(ctx) >= ctx_len:
                    padded.append(ctx[-ctx_len:])
                else:
                    padded.append(np.pad(ctx, (ctx_len - len(ctx), 0), mode='constant'))
            self.contexts = torch.tensor(np.array(padded, dtype=np.float32)).unsqueeze(-1)
            self.targets = torch.tensor(np.array(targets, dtype=np.float32)).unsqueeze(-1)

        def __len__(self):
            return len(self.contexts)

        def __getitem__(self, idx):
            return {
                "past_values": self.contexts[idx],
                "future_values": self.targets[idx],
                "freq_token": torch.tensor(0, dtype=torch.long),  # 0 = hourly
            }

    for pred_len in PREDICTION_LENGTHS:
        print(f"\n{'=' * 60}")
        print(f"Fine-tuning TTM-R2 for {pred_len}h horizon")
        print(f"{'=' * 60}")

        contexts, targets = build_datasets(pred_len)
        print(f"Training samples: {len(contexts)}")

        model = get_model(
            MODEL_ID,
            context_length=512,  # Use TTM's 512-context variant
            prediction_length=pred_len,
        )
        ttm_ctx_len = model.config.context_length

        # Split 90/10 for train/val, pad to model's context length
        n_val = max(1, len(contexts) // 10)
        train_ds = SolarDataset(contexts[:-n_val], targets[:-n_val], ttm_ctx_len)
        val_ds = SolarDataset(contexts[-n_val:], targets[-n_val:], ttm_ctx_len)

        training_args = TrainingArguments(
            output_dir=str(output_path / f"{pred_len}h"),
            num_train_epochs=MAX_EPOCHS,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            learning_rate=1e-3,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=50,
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        )

        print(f"\nStarting training...")
        trainer.train()

        # Save the best model
        save_path = output_path / f"ttm_{pred_len}h"
        trainer.save_model(str(save_path))
        print(f"Model saved to {save_path}")

    print(f"\nFine-tuning complete! Models saved to {output_path}")


if __name__ == "__main__":
    log = create_log("finetune_ttm", {"model_id": MODEL_ID, "max_epochs": MAX_EPOCHS})
    log_step(log, "finetune_start")
    finetune()
    log_step(log, "finetune_done")
    save_log(log)
