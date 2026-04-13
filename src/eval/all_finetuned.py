"""Evaluate ALL fine-tuned models vs zero-shot vs baselines.

Generates comprehensive comparison with:
- MAE, RMSE, MASE, CRPS metrics
- Diebold-Mariano statistical significance tests
- Results saved to CSV
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.seed import set_seed
from utils.metrics import compute_all_metrics, pairwise_dm_tests
from utils.logger import create_log, log_step, log_metrics, log_error, save_log, save_summary, get_device
from utils.eval_utils import load_windows, get_season, persistence_forecast
from config import MODELS_DIR, TABLES_DIR, PREDICTION_LENGTHS, CONTEXT_LENGTH

set_seed()

RESULTS_DIR = TABLES_DIR


# ── Model evaluators ────────────────────────────────────────────────────────

def eval_chronos(model_path, contexts, pred_len):
    """Chronos inference returning (medians, samples)."""
    from chronos import ChronosPipeline
    from utils.logger import get_device

    device = get_device()
    model_str = str(model_path)
    local_files_only = not model_str.startswith(("amazon/", "google/", "Salesforce/"))
    pipeline = ChronosPipeline.from_pretrained(
        model_str, device_map=device, dtype=torch.float32,
        local_files_only=local_files_only,
    )

    all_medians, all_samples = [], []
    for i in range(0, len(contexts), 32):
        batch = [torch.tensor(ctx, dtype=torch.float32) for ctx in contexts[i:i+32]]
        forecast = pipeline.predict(batch, prediction_length=pred_len, num_samples=20)
        samples_np = forecast.numpy()
        all_medians.append(np.median(samples_np, axis=1))
        all_samples.append(samples_np)

    del pipeline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    medians = np.clip(np.concatenate(all_medians), 0, None)
    samples = np.clip(np.concatenate(all_samples), 0, None)
    return medians, samples


def eval_chronos2(model_path, contexts, pred_len):
    """Chronos-2 inference returning (medians, samples)."""
    from chronos import Chronos2Pipeline
    from utils.logger import get_device

    device = get_device()
    pipeline = Chronos2Pipeline.from_pretrained(str(model_path), device_map=device)

    all_medians, all_samples = [], []
    for i in range(0, len(contexts), 32):
        batch = [torch.tensor(ctx, dtype=torch.float32) for ctx in contexts[i:i+32]]
        forecast = pipeline.predict(batch, prediction_length=pred_len, num_samples=20)
        samples_np = forecast.numpy()
        all_medians.append(np.median(samples_np, axis=1))
        all_samples.append(samples_np)

    del pipeline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    medians = np.clip(np.concatenate(all_medians), 0, None)
    samples = np.clip(np.concatenate(all_samples), 0, None)
    return medians, samples


def eval_timesfm(model_path, contexts, pred_len):
    """TimesFM 2.5 inference."""
    import timesfm

    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(model_path))
    tfm.compile(timesfm.ForecastConfig(
        max_context=CONTEXT_LENGTH,
        max_horizon=pred_len,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    ))

    context_array = np.array(contexts)
    point_forecasts, _ = tfm.forecast(
        horizon=pred_len,
        inputs=[ctx for ctx in context_array],
    )
    predictions = np.clip(np.array(point_forecasts)[:, :pred_len], 0, None)
    del tfm
    return predictions, None


def eval_moirai2(model_path, contexts, pred_len):
    """Moirai 2.0 inference."""
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    from gluonts.dataset.common import ListDataset

    module = Moirai2Module.from_pretrained(str(model_path))
    forecast_module = Moirai2Forecast(
        module=module,
        prediction_length=pred_len,
        context_length=CONTEXT_LENGTH,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = forecast_module.create_predictor(batch_size=32)

    dataset = ListDataset(
        [{"start": pd.Timestamp("2025-01-01"), "target": ctx} for ctx in contexts],
        freq="h",
    )

    all_medians, all_samples = [], []
    for forecast in predictor.predict(dataset):
        s = forecast.samples[:, :pred_len]
        all_samples.append(s)
        all_medians.append(np.median(s, axis=0))

    del module, predictor
    medians = np.clip(np.array(all_medians), 0, None)
    samples = np.clip(np.array(all_samples), 0, None)
    return medians, samples


def eval_ttm(model_path, contexts, pred_len):
    """TTM-R2 inference."""
    from tsfm_public.toolkit.get_model import get_model
    from utils.logger import get_device

    device = get_device()

    # Check for fine-tuned model
    ft_path = MODELS_DIR / "ft-ttm-r2" / f"ttm_{pred_len}h"
    if isinstance(model_path, Path) or (ft_path.exists() and "FT" in str(model_path)):
        model = get_model(str(ft_path), context_length=CONTEXT_LENGTH, prediction_length=pred_len)
    else:
        model = get_model(str(model_path), context_length=CONTEXT_LENGTH, prediction_length=pred_len)

    model = model.to(device).eval()

    predictions = []
    for i in range(0, len(contexts), 64):
        batch = np.array(contexts[i:i+64], dtype=np.float32)
        batch_tensor = torch.tensor(batch).unsqueeze(-1).to(device)
        with torch.no_grad():
            output = model(batch_tensor)
            if hasattr(output, 'prediction_outputs'):
                pred = output.prediction_outputs.squeeze(-1).cpu().numpy()
            else:
                pred = output.squeeze(-1).cpu().numpy()
        predictions.append(pred[:, :pred_len])

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    preds = np.clip(np.concatenate(predictions), 0, None)
    return preds, None


def eval_baseline_preds(pred_len, model_name):
    """Load pre-computed baseline predictions."""
    safe_name = model_name.lower()
    pred_path = RESULTS_DIR.parent / f"preds_{safe_name}_{pred_len}h.npy"
    if pred_path.exists():
        return np.load(pred_path), None
    return None, None


# ── Main ────────────────────────────────────────────────────────────────────

def build_model_registry():
    """Build list of all models to evaluate."""
    models = []

    # Baselines (from saved predictions)
    models.append(("Persistence", "baseline", None))
    models.append(("XGBoost", "baseline", None))
    models.append(("LSTM", "baseline", None))

    # Chronos zero-shot + fine-tuned
    models.append(("Chronos-T5-Small (ZS)", "chronos", "amazon/chronos-t5-small"))
    if (MODELS_DIR / "ft-chronos-t5-small" / "config.json").exists():
        models.append(("Chronos-T5-Small (FT)", "chronos", MODELS_DIR / "ft-chronos-t5-small"))
    models.append(("Chronos-T5-Base (ZS)", "chronos", "amazon/chronos-t5-base"))
    if (MODELS_DIR / "ft-chronos-t5-base" / "config.json").exists():
        models.append(("Chronos-T5-Base (FT)", "chronos", MODELS_DIR / "ft-chronos-t5-base"))

    # Chronos-2 zero-shot
    models.append(("Chronos-2 (ZS)", "chronos2", "amazon/chronos-2"))

    # TimesFM 2.5 zero-shot
    models.append(("TimesFM-2.5 (ZS)", "timesfm", "google/timesfm-2.5-200m-pytorch"))

    # Moirai 2.0 zero-shot
    models.append(("Moirai-2.0-Small (ZS)", "moirai2", "Salesforce/moirai-2.0-R-small"))

    # TTM-R2 zero-shot + fine-tuned
    models.append(("TTM-R2 (ZS)", "ttm", "ibm-granite/granite-timeseries-ttm-r2"))
    for pred_len in PREDICTION_LENGTHS:
        ft_path = MODELS_DIR / "ft-ttm-r2" / f"ttm_{pred_len}h"
        if ft_path.exists():
            models.append(("TTM-R2 (FT)", "ttm", ft_path))
            break

    return models


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    models = build_model_registry()
    log = create_log("eval_all_finetuned", {
        "models": [name for name, _, _ in models],
        "prediction_lengths": PREDICTION_LENGTHS,
    })
    all_results = []

    print(f"Models to evaluate: {len(models)}")
    for name, _, _ in models:
        print(f"  - {name}")

    for pred_len in PREDICTION_LENGTHS:
        print(f"\n{'=' * 70}")
        print(f"Prediction horizon: {pred_len}h")
        print(f"{'=' * 70}")

        contexts, targets, timestamps = load_windows("test", pred_len)
        timestamps = pd.DatetimeIndex(timestamps)
        seasons = np.array([get_season(m) for m in timestamps.month])

        model_errors = {}

        for model_label, family, model_path in models:
            print(f"\n  {model_label}...")
            log_step(log, f"eval_{model_label}_{pred_len}h")

            try:
                if family == "baseline":
                    if model_label == "Persistence":
                        preds = persistence_forecast(contexts, pred_len)
                        samples = None
                    else:
                        preds, samples = eval_baseline_preds(pred_len, model_label)
                        if preds is None:
                            print(f"    SKIPPED: no saved predictions")
                            continue
                elif family == "chronos":
                    preds, samples = eval_chronos(model_path, contexts, pred_len)
                elif family == "chronos2":
                    preds, samples = eval_chronos2(model_path, contexts, pred_len)
                elif family == "timesfm":
                    preds, samples = eval_timesfm(model_path, contexts, pred_len)
                elif family == "moirai2":
                    preds, samples = eval_moirai2(model_path, contexts, pred_len)
                elif family == "ttm":
                    preds, samples = eval_ttm(model_path, contexts, pred_len)
                else:
                    continue
            except Exception as e:
                print(f"    FAILED: {e}")
                log_error(log, f"{model_label} {pred_len}h: {e}")
                continue

            model_errors[model_label] = np.abs(preds - targets)

            # Save predictions
            safe_name = model_label.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            np.save(RESULTS_DIR.parent / f"preds_{safe_name}_{pred_len}h.npy", preds)

            for season_label in ["all", "dry", "wet"]:
                mask = np.ones(len(targets), dtype=bool) if season_label == "all" else (seasons == season_label)
                m = compute_all_metrics(
                    preds[mask], targets[mask],
                    samples=samples[mask] if samples is not None else None,
                )
                all_results.append({
                    "model": model_label,
                    "horizon": f"{pred_len}h",
                    "season": season_label,
                    **m,
                })

            overall = compute_all_metrics(preds, targets, samples=samples)
            log_metrics(log, model_label, f"{pred_len}h", "all", overall)
            crps_str = f", CRPS: {overall['CRPS']:.2f}" if "CRPS" in overall else ""
            print(f"    MAE: {overall['MAE']:.2f}, RMSE: {overall['RMSE']:.2f}, "
                  f"MASE: {overall['MASE']:.3f}{crps_str}")

        # DM tests
        if len(model_errors) > 1:
            print(f"\n  Diebold-Mariano Tests ({pred_len}h):")
            model_names = list(model_errors.keys())
            dm_results = pairwise_dm_tests(model_errors, model_names, horizon=pred_len)
            for r in dm_results:
                print(f"    {r['model_A']} vs {r['model_B']}: p={r['p_value']:.4f} {r['significance']}")
            dm_df = pd.DataFrame(dm_results)
            dm_df["horizon"] = f"{pred_len}h"
            dm_df.to_csv(RESULTS_DIR / f"dm_tests_all_{pred_len}h.csv", index=False)

    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "all_models_comparison.csv", index=False)
    save_summary(all_results, "all_models_comparison")
    save_log(log)
    print(f"\n{'=' * 70}")
    print("ALL RESULTS")
    print(f"{'=' * 70}")
    print(results_df[results_df["season"] == "all"].to_string(index=False))
    print(f"\nSaved to results/tables/all_models_comparison.csv")


if __name__ == "__main__":
    main()
