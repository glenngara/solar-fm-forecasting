"""Zero-shot evaluation of time series foundation models on tropical solar irradiance.

Evaluates multiple FMs on the test set without any fine-tuning:
- Chronos-2 (Amazon) — 120M params, encoder-only
- Chronos-T5 Small/Base (Amazon) — for fine-tuning comparison
- TimesFM 2.5 (Google) — 200M params
- Moirai 2.0 (Salesforce) — decoder-only
- TTM-R2 (IBM) — MLP-Mixer, ~1M params
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.seed import set_seed
from utils.logger import create_log, log_step, log_metrics, log_error, save_log, save_summary, get_device
from utils.eval_utils import load_windows, get_season, compute_metrics, persistence_forecast
from config import TABLES_DIR, PREDICTION_LENGTHS, CONTEXT_LENGTH, FM_REGISTRY

set_seed()

RESULTS_DIR = TABLES_DIR


# ── Chronos-2 ────────────────────────────────────────────────────────────────

def evaluate_chronos2(model_id, contexts, pred_len):
    """Run Chronos-2 zero-shot inference (new encoder-only architecture)."""
    from chronos import Chronos2Pipeline

    device = get_device()
    pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=device)

    all_preds = []
    batch_size = 32
    for i in range(0, len(contexts), batch_size):
        batch = [torch.tensor(ctx, dtype=torch.float32) for ctx in contexts[i : i + batch_size]]
        forecast = pipeline.predict(batch, prediction_length=pred_len)
        # Chronos-2 output shape: (batch, num_series, num_samples, pred_len)
        # Convert to numpy and take median across samples
        f = forecast.numpy() if hasattr(forecast, 'numpy') else np.array(forecast)
        # Squeeze extra dims and take median across sample dim
        while f.ndim > 2:
            if f.shape[1] == 1:
                f = f.squeeze(1)  # remove num_series dim
            else:
                f = np.median(f, axis=-2)  # median across samples
        all_preds.append(f)

    del pipeline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.clip(np.concatenate(all_preds, axis=0), 0, None)


# ── Chronos v1 (T5-based, for fine-tuning comparison) ────────────────────────

def evaluate_chronos(model_id, contexts, pred_len):
    """Run Chronos v1 (T5) zero-shot inference."""
    from chronos import ChronosPipeline

    device = get_device()
    pipeline = ChronosPipeline.from_pretrained(
        model_id, device_map=device, dtype=torch.float32,
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


# ── TimesFM 2.5 ─────────────────────────────────────────────────────────────

def evaluate_timesfm(model_id, contexts, pred_len):
    """Run TimesFM 2.5 zero-shot inference."""
    import timesfm

    # TimesFM 2.5 — try multiple instantiation methods
    tfm = None

    # Method 1: No-arg constructor (auto-downloads from default repo)
    try:
        tfm = timesfm.TimesFM_2p5_200M_torch()
    except Exception:
        pass

    # Method 2: Print available init args and try with common patterns
    if tfm is None:
        import inspect
        sig = inspect.signature(timesfm.TimesFM_2p5_200M_torch.__init__)
        params = list(sig.parameters.keys())
        print(f"  TimesFM init params: {params}")

        # Try with repo_id
        for kwarg in ["repo_id", "model_id", "checkpoint", "pretrained_model_name_or_path"]:
            if kwarg in params:
                try:
                    tfm = timesfm.TimesFM_2p5_200M_torch(**{kwarg: model_id})
                    break
                except Exception:
                    continue

    if tfm is None:
        raise ImportError(f"Could not instantiate TimesFM_2p5_200M_torch")

    # Compile with device specification
    device = get_device()
    tfm.compile(timesfm.ForecastConfig(
        max_context=CONTEXT_LENGTH,
        max_horizon=pred_len,
        backend=device,
    ))

    context_array = np.array(contexts)
    # Try different forecast APIs
    try:
        point_forecasts, _ = tfm.forecast(
            inputs=list(context_array),
            horizon=pred_len,
        )
    except TypeError:
        frequency_input = [0] * len(contexts)
        point_forecasts, _ = tfm.forecast(context_array, freq=frequency_input)

    predictions = np.array(point_forecasts)[:, :pred_len]

    del tfm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.clip(predictions, 0, None)


# ── Moirai 2.0 ──────────────────────────────────────────────────────────────

def evaluate_moirai2(model_id, contexts, pred_len):
    """Run Moirai 2.0 zero-shot inference."""
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    from gluonts.dataset.common import ListDataset

    module = Moirai2Module.from_pretrained(model_id)
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

    predictions = []
    for forecast in predictor.predict(dataset):
        # Moirai 2.0 returns QuantileForecast (no .samples), use .quantile() or ._forecast_array
        if hasattr(forecast, 'samples'):
            pred = np.median(forecast.samples, axis=0)
        elif hasattr(forecast, '_forecast_array'):
            # QuantileForecast: median is the middle quantile
            pred = np.median(forecast._forecast_array, axis=0)
        elif hasattr(forecast, 'quantile'):
            pred = forecast.quantile(0.5)
        else:
            pred = forecast.mean
        predictions.append(pred[:pred_len])

    del module, predictor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.clip(np.array(predictions), 0, None)


# ── TTM-R2 (IBM Granite) ────────────────────────────────────────────────────

def evaluate_ttm(model_id, contexts, pred_len):
    """Run TTM-R2 zero-shot inference with rolling prediction for longer horizons."""
    from tsfm_public.toolkit.get_model import get_model

    device = get_device()

    # TTM-R2 supports context lengths: 90, 512, 1536.
    # Use 512-step variant so it can accept our full 168h context.
    # Max prediction is ~30 steps — use rolling prediction for longer horizons.
    TTM_CTX = 512
    ttm_pred_len = min(pred_len, 30)
    model = get_model(
        model_id,
        context_length=TTM_CTX,
        prediction_length=ttm_pred_len,
    )
    model = model.to(device).eval()
    ttm_ctx_len = model.config.context_length

    predictions = []
    batch_size = 64
    for i in range(0, len(contexts), batch_size):
        batch_contexts = contexts[i : i + batch_size]

        def _pad_context(ctx, target_len):
            """Pad context to target_len with zeros on the left if shorter."""
            if len(ctx) >= target_len:
                return ctx[-target_len:]
            return np.pad(ctx, (target_len - len(ctx), 0), mode='constant', constant_values=0)

        if pred_len <= ttm_pred_len:
            # Single-step prediction
            batch = np.array([_pad_context(ctx, ttm_ctx_len) for ctx in batch_contexts], dtype=np.float32)
            past_values = torch.tensor(batch).unsqueeze(-1).to(device)
            freq_token = torch.zeros(len(batch), dtype=torch.long, device=device)

            with torch.no_grad():
                try:
                    output = model(past_values=past_values, freq_token=freq_token)
                except TypeError:
                    output = model(past_values=past_values)

                if hasattr(output, 'prediction_outputs'):
                    pred = output.prediction_outputs.squeeze(-1).cpu().numpy()
                else:
                    pred = output[0].squeeze(-1).cpu().numpy() if isinstance(output, tuple) else output.squeeze(-1).cpu().numpy()

            predictions.append(pred[:, :pred_len])
        else:
            # Rolling prediction for longer horizons
            batch_preds = []
            for ctx in batch_contexts:
                rolling_ctx = list(_pad_context(ctx, ttm_ctx_len))
                full_pred = []
                remaining = pred_len
                while remaining > 0:
                    inp = np.array([rolling_ctx[-ttm_ctx_len:]], dtype=np.float32)
                    past_values = torch.tensor(inp).unsqueeze(-1).to(device)
                    freq_token = torch.zeros(1, dtype=torch.long, device=device)
                    with torch.no_grad():
                        try:
                            output = model(past_values=past_values, freq_token=freq_token)
                        except TypeError:
                            output = model(past_values=past_values)
                        if hasattr(output, 'prediction_outputs'):
                            step_pred = output.prediction_outputs.squeeze().cpu().numpy()
                        else:
                            step_pred = output[0].squeeze().cpu().numpy() if isinstance(output, tuple) else output.squeeze().cpu().numpy()
                    take = min(len(step_pred), remaining)
                    full_pred.extend(step_pred[:take])
                    rolling_ctx.extend(step_pred[:take])
                    remaining -= take
                batch_preds.append(full_pred[:pred_len])
            predictions.append(np.array(batch_preds))

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return np.clip(np.concatenate(predictions, axis=0), 0, None)


# ── Dispatcher ───────────────────────────────────────────────────────────────

EVALUATORS = {
    "chronos2": evaluate_chronos2,
    "chronos": evaluate_chronos,
    "timesfm": evaluate_timesfm,
    "moirai2": evaluate_moirai2,
    "ttm": evaluate_ttm,
}


def evaluate_fm(display_name, family, model_id, contexts, pred_len):
    """Dispatch to the correct evaluator, with error handling."""
    evaluator = EVALUATORS.get(family)
    if evaluator is None:
        print(f"  SKIPPED {display_name}: unknown model family '{family}'")
        return None
    try:
        return evaluator(model_id, contexts, pred_len)
    except ImportError as e:
        print(f"  SKIPPED {display_name}: missing dependency — {e}")
        print(f"  Install the required package and re-run.")
        return None
    except Exception as e:
        print(f"  FAILED {display_name}: {e}")
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = create_log("eval_zero_shot", {
        "models": [name for name, _, _ in FM_REGISTRY],
        "prediction_lengths": PREDICTION_LENGTHS,
        "context_length": CONTEXT_LENGTH,
        "num_samples": 20,
    })
    all_results = []

    for pred_len in PREDICTION_LENGTHS:
        print(f"\n{'=' * 60}")
        print(f"Prediction horizon: {pred_len}h")
        print(f"{'=' * 60}")

        contexts, targets, timestamps = load_windows("test", pred_len)
        timestamps = pd.DatetimeIndex(timestamps)
        months = timestamps.month
        seasons = np.array([get_season(m) for m in months])

        # Persistence baseline
        print("Evaluating persistence baseline...")
        persist_preds = persistence_forecast(contexts, pred_len)
        persist_metrics = compute_metrics(persist_preds, targets)
        all_results.append({
            "model": "Persistence",
            "horizon": f"{pred_len}h",
            "season": "all",
            **persist_metrics,
        })
        for season in ["dry", "wet"]:
            mask = seasons == season
            m = compute_metrics(persist_preds[mask], targets[mask])
            all_results.append({
                "model": "Persistence",
                "horizon": f"{pred_len}h",
                "season": season,
                **m,
            })
        print(f"  Persistence MAE: {persist_metrics['MAE']:.2f}")
        log_step(log, "persistence", {"horizon": pred_len, "MAE": persist_metrics["MAE"]})

        # Foundation models
        for display_name, family, model_id in FM_REGISTRY:
            print(f"\nEvaluating {display_name} ({model_id})...")
            log_step(log, f"start_{display_name}", {"horizon": pred_len})

            preds = evaluate_fm(display_name, family, model_id, contexts, pred_len)

            if preds is None:
                log_error(log, f"{display_name} failed for {pred_len}h")
                continue

            # Overall metrics
            metrics = compute_metrics(preds, targets)
            skill = 1 - metrics["MAE"] / persist_metrics["MAE"]
            result_entry = {
                "model": display_name,
                "horizon": f"{pred_len}h",
                "season": "all",
                **metrics,
                "skill_score": skill,
            }
            all_results.append(result_entry)
            log_metrics(log, display_name, f"{pred_len}h", "all", {**metrics, "skill_score": skill})
            print(f"  {display_name} MAE: {metrics['MAE']:.2f} (skill: {skill:.3f})")

            # Seasonal breakdown
            for season in ["dry", "wet"]:
                mask = seasons == season
                m = compute_metrics(preds[mask], targets[mask])
                persist_m = compute_metrics(persist_preds[mask], targets[mask])
                s = 1 - m["MAE"] / persist_m["MAE"]
                all_results.append({
                    "model": display_name,
                    "horizon": f"{pred_len}h",
                    "season": season,
                    **m,
                    "skill_score": s,
                })
                log_metrics(log, display_name, f"{pred_len}h", season, {**m, "skill_score": s})
                print(f"    {season}: MAE={m['MAE']:.2f}, skill={s:.3f}")

            # Save predictions
            safe_name = display_name.lower().replace("-", "_").replace(" ", "_")
            np.save(RESULTS_DIR.parent / f"preds_{safe_name}_{pred_len}h.npy", preds)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "zero_shot_results.csv", index=False)
    save_summary(all_results, "zero_shot_results")
    save_log(log)

    print(f"\n{'=' * 60}")
    print("Results saved to results/tables/zero_shot_results.csv")
    print(f"{'=' * 60}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
