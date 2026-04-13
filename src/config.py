"""Central configuration for all pipeline scripts.

Single source of truth for constants, paths, and model registry.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

# ── Data ─────────────────────────────────────────────────────────────────────
TARGET_COL = "ALLSKY_SFC_SW_DWN"
RAW_FILENAME = "nasa_power_laguna_de_bay_2020_2025.csv"

# ── Forecasting ──────────────────────────────────────────────────────────────
PREDICTION_LENGTHS = [24, 72]
CONTEXT_LENGTH = 168  # 7 days of hourly data
SEED = 42

# ── Foundation Models (Zero-Shot Evaluation) ─────────────────────────────────
FM_REGISTRY = [
    # (display_name, model_family, huggingface_id)
    ("Chronos-2", "chronos2", "amazon/chronos-2"),
    ("Chronos-T5-Small", "chronos", "amazon/chronos-t5-small"),
    ("Chronos-T5-Base", "chronos", "amazon/chronos-t5-base"),
    ("TimesFM-2.5", "timesfm", "google/timesfm-2.5-200m-pytorch"),
    ("Moirai-2.0-Small", "moirai2", "Salesforce/moirai-2.0-R-small"),
    ("TTM-R2", "ttm", "ibm-granite/granite-timeseries-ttm-r2"),
]

# ── Fine-tuning Targets ─────────────────────────────────────────────────────
# Chronos v1 — uses HuggingFace Trainer with Chronos tokenizer (proven, works)
FINETUNE_CHRONOS = [
    ("amazon/chronos-t5-small", "ft-chronos-t5-small"),
    ("amazon/chronos-t5-base", "ft-chronos-t5-base"),
]

# TTM-R2 — uses standard HuggingFace Trainer (lightweight, ~1M params)
FINETUNE_TTM = ("ibm-granite/granite-timeseries-ttm-r2", "ft-ttm-r2")
