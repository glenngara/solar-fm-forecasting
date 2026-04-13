# Evaluating Time Series Foundation Models for Tropical Solar Irradiance Forecasting

A case study on Laguna de Bay, Philippines — targeting PREE 2026 (IEEE Xplore).

## Research Question

Can pre-trained time series foundation models (Chronos-2, TimesFM 2.5, Moirai 2.0, TTM-R2) accurately forecast solar irradiance in tropical monsoon climates? Does fine-tuning on local data improve performance, especially during the wet season?

## Reproducibility

All experiments use a **global seed of 42** (set via `src/utils/seed.py`) ensuring deterministic results for NumPy, PyTorch, Python random, and PyTorch Lightning.

All output directories (`data/`, `results/`, `models/`) are auto-created by the scripts — no manual setup needed.

### Prerequisites

- Python 3.11+
- Git
- NVIDIA GPU with CUDA 12.1+ (recommended for fine-tuning)
- ~4GB disk for model downloads

### Step 1: Clone the repository

```bash
git clone https://github.com/<your-username>/solar-fm-forecasting.git
cd solar-fm-forecasting
```

### Step 2: Create virtual environment

**macOS/Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install git-only packages

These models are not on PyPI and must be installed from GitHub:

```bash
pip install "chronos-forecasting[training]>=2.0"
pip install granite-tsfm
pip install "uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts.git"
pip install git+https://github.com/google-research/timesfm.git
```

### Step 5: Install PyTorch with CUDA

Install PyTorch **last** to prevent other packages from overwriting it with a CPU-only version.

**CUDA — Stable (most GPUs up to RTX 40 series):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

**CUDA — Nightly (required for RTX 50 series / Blackwell GPUs):**
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
```

**macOS (Apple Silicon):**
```bash
pip install torch torchvision --force-reinstall
```

**CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --force-reinstall
```

### Step 6: Verify installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
python -c "from chronos import Chronos2Pipeline; print('Chronos-2: OK')"
python -c "from tsfm_public.toolkit.get_model import get_model; print('TTM-R2: OK')"
python -c "import timesfm; print('TimesFM: OK')"
python -c "from uni2ts.model.moirai2 import Moirai2Module; print('Moirai-2: OK')"
python -c "import gluonts; print('GluonTS: OK')"
```

### Alternative: Use setup scripts

```bash
# macOS/Linux
chmod +x setup.sh && ./setup.sh

# Windows
setup.bat
```

### Step 7: Run the pipeline

```bash
# Full pipeline
make all

# Or use the Python orchestrator
python src/run_all.py
```

To reproduce all results from scratch:
```bash
make clean     # Remove all generated outputs
make all       # Re-run entire pipeline
```

## Pipeline

| Step | Command | Script | Description |
|------|---------|--------|-------------|
| 1 | `make data` | `data/download_nasa_power.py` | Download NASA POWER data (2020-2025) |
| 2 | `make prepare` | `data/prepare_data.py` | Train/val/test splits + forecast windows |
| 3 | `make eda` | `figures/eda.py` | Exploratory data analysis figures |
| 4 | `make zero-shot` | `eval/zero_shot.py` | Zero-shot FM evaluation (6 models) |
| 5 | `make baselines` | `eval/baselines.py` | Traditional baselines (XGBoost, LSTM) |
| 6 | `make finetune-chronos` | `finetune/chronos_ft.py` | Fine-tune Chronos Small + Base |
| 7 | `make finetune-ttm` | `finetune/ttm_ft.py` | Fine-tune TTM-R2 |
| 8 | `make eval-finetuned` | `eval/finetuned.py` | Fine-tuned Chronos vs zero-shot |
| 9 | `make eval-all` | `eval/all_finetuned.py` | Full comparison + CRPS + DM tests |
| 10 | `make eval-ablation` | `eval/ablation.py` | Ablation: training steps vs performance |
| 11 | `make data-efficiency` | `experiments/data_efficiency.py` | Training data size experiment |
| 12 | `make sensitivity` | `experiments/sensitivity_analysis.py` | Hyperparameter sweep (27 configs) |
| 13 | `make figures` | `figures/generate.py` | Generate all paper figures |

Use `make finetune-all` to run steps 6-7 together, or `make all` for the full pipeline.

### Selective Execution

```bash
# List all steps
python src/run_all.py --list

# Run specific steps
python src/run_all.py --steps 4,6-9,13

# Run from step 6 onwards
python src/run_all.py --from 6

# Continue past failures
python src/run_all.py --steps 6-9 --no-stop
```

Or run step-by-step via the notebook: `notebooks/pipeline.ipynb`

## Data

- **Source:** NASA POWER API (free, no registration)
- **Location:** Laguna de Bay, Philippines (14.3833N, 121.2500E)
- **Period:** 2020-01-01 to 2025-12-31 (hourly)
- **Records:** 52,608
- **Parameters:** Solar irradiance (ALLSKY_SFC_SW_DWN), temperature, humidity, wind speed, cloud cover
- **Splits:** Train 2020-2023, Val 2024, Test 2025

Data is downloaded automatically by the pipeline via `make data`. No manual download needed.

## Models

**Foundation Models:**

| Model | Provider | Params | Architecture | Fine-tuned | Year |
|-------|----------|--------|-------------|------------|------|
| Chronos-2 | Amazon | 120M | Encoder-only | No (zero-shot) | 2025 |
| Chronos-T5 Small | Amazon | ~60M | T5 enc-dec | Yes | 2024 |
| Chronos-T5 Base | Amazon | ~200M | T5 enc-dec | Yes | 2024 |
| TimesFM 2.5 | Google | 200M | Patched decoder | No (zero-shot) | 2026 |
| Moirai 2.0 Small | Salesforce | ~14M | Decoder-only | No (zero-shot) | 2025 |
| TTM-R2 | IBM | ~1M | MLP-Mixer | Yes | 2025 |

**Baselines:**

| Model | Type | Description |
|-------|------|-------------|
| Persistence | Naive | Repeat last day's pattern |
| XGBoost | ML | Engineered features, per-step models |
| LSTM | DL | 2-layer, 64 hidden units |

## Evaluation Metrics

| Metric | Type | Purpose |
|--------|------|---------|
| MAE | Point | Mean absolute error |
| RMSE | Point | Root mean squared error |
| MASE | Point (scaled) | Scale-free comparison across datasets |
| CRPS | Probabilistic | Calibration of full forecast distribution |
| Diebold-Mariano | Statistical | Significance of pairwise model differences |

All metrics are computed overall and by season (dry: Dec-May, wet: Jun-Nov) for both 24h and 72h forecast horizons.

## Project Structure

```
solar-fm-forecasting/
├── Makefile                 # Reproducible pipeline (make all)
├── requirements.txt         # Python dependencies (pinned)
├── setup.sh / setup.bat     # Environment setup scripts
├── README.md
├── data/                    # Auto-created by pipeline
│   ├── raw/                 # NASA POWER CSV
│   └── processed/           # Train/val/test splits + forecast windows
├── models/                  # Auto-created: fine-tuned checkpoints
│   ├── ft-chronos-t5-small/
│   ├── ft-chronos-t5-base/
│   └── ft-ttm-r2/
├── notebooks/
│   └── pipeline.ipynb       # Interactive pipeline + results viewer
├── results/                 # Auto-created by pipeline
│   ├── figures/             # EDA + paper figures
│   ├── tables/              # Metrics CSVs + DM test results
│   └── logs/                # JSON/Markdown experiment logs (overwritten on re-run)
└── src/
    ├── config.py            # Central configuration (paths, constants, model registry)
    ├── run_all.py           # Pipeline orchestrator
    ├── utils/
    │   ├── seed.py          # Global seed (42) for reproducibility
    │   ├── metrics.py       # Shared metrics (MAE, RMSE, MASE, CRPS, DM)
    │   ├── logger.py        # Structured logging utility
    │   └── eval_utils.py    # Shared eval functions (load_windows, get_season, etc.)
    ├── data/
    │   ├── download_nasa_power.py
    │   └── prepare_data.py
    ├── eval/
    │   ├── zero_shot.py     # Zero-shot FM evaluation (6 models)
    │   ├── baselines.py     # XGBoost + LSTM baselines
    │   ├── finetuned.py     # Fine-tuned Chronos vs zero-shot
    │   ├── all_finetuned.py # Full comparison + CRPS + DM tests
    │   └── ablation.py      # Training steps ablation study
    ├── finetune/
    │   ├── chronos_ft.py    # Fine-tune Chronos Small + Base
    │   └── ttm_ft.py        # Fine-tune TTM-R2
    ├── experiments/
    │   ├── data_efficiency.py
    │   └── sensitivity_analysis.py
    └── figures/
        ├── eda.py           # Exploratory data analysis figures
        └── generate.py      # Publication figures from results
```

## Authors

- Glenn Paul Gara
- Gretchie Gara
- Marvin Santillan
