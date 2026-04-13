"""Traditional baseline models for solar irradiance forecasting.

Evaluates XGBoost and LSTM against the same test windows used for FM evaluation.
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.seed import set_seed
from utils.logger import create_log, log_step, save_log, save_summary, get_device
from utils.eval_utils import load_windows, get_season, compute_metrics
from config import PROCESSED_DIR, TABLES_DIR, PREDICTION_LENGTHS, CONTEXT_LENGTH

set_seed()

RESULTS_DIR = TABLES_DIR


# ── XGBoost ──────────────────────────────────────────────────────────────────

def create_xgb_features(context):
    """Engineer features from a context window for XGBoost.

    Features: hourly values of last 3 days, daily means of last 7 days,
    hour-of-day encoding, day-of-year stats.
    """
    features = []

    # Last 72 hours raw (3 days)
    features.extend(context[-72:])

    # Daily means for each of the 7 days
    for d in range(7):
        start = d * 24
        end = start + 24
        day_vals = context[start:end]
        features.append(np.mean(day_vals))
        features.append(np.std(day_vals))
        features.append(np.max(day_vals))

    # Daytime-only stats (hours 6-18) for last 3 days
    for d in range(4, 7):  # last 3 of the 7 days
        start = d * 24 + 6
        end = d * 24 + 18
        day_vals = context[start:end]
        features.append(np.mean(day_vals))
        features.append(np.max(day_vals))

    return np.array(features)


def train_and_eval_xgboost(pred_len):
    """Train XGBoost: one model per forecast step (direct multi-step)."""
    print(f"\n  Training XGBoost for {pred_len}h horizon...")

    # Load training windows
    train_contexts, train_targets, _ = load_windows("val", pred_len)
    test_contexts, test_targets, test_timestamps = load_windows("test", pred_len)

    # Also create training windows from the training set
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv", index_col="timestamp", parse_dates=True)
    train_values = train_df["ALLSKY_SFC_SW_DWN"].interpolate(limit=3).dropna().values

    # Create windows from training data
    stride = 24
    extra_contexts, extra_targets = [], []
    for i in range(0, len(train_values) - CONTEXT_LENGTH - pred_len + 1, stride):
        ctx = train_values[i : i + CONTEXT_LENGTH]
        tgt = train_values[i + CONTEXT_LENGTH : i + CONTEXT_LENGTH + pred_len]
        if not (np.isnan(ctx).any() or np.isnan(tgt).any()):
            extra_contexts.append(ctx)
            extra_targets.append(tgt)

    all_train_ctx = np.array(extra_contexts + list(train_contexts))
    all_train_tgt = np.array(extra_targets + list(train_targets))

    # Build feature matrices
    X_train = np.array([create_xgb_features(c) for c in all_train_ctx])
    X_test = np.array([create_xgb_features(c) for c in test_contexts])

    # Train one model per forecast step
    predictions = np.zeros((len(test_contexts), pred_len))
    for step in range(pred_len):
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, all_train_tgt[:, step])
        predictions[:, step] = model.predict(X_test)

        if (step + 1) % 24 == 0:
            print(f"    Step {step + 1}/{pred_len} done")

    predictions = np.clip(predictions, 0, None)
    return predictions, test_targets, test_timestamps


# ── LSTM ─────────────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=24):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out


def train_and_eval_lstm(pred_len):
    """Train LSTM on training data, evaluate on test."""
    print(f"\n  Training LSTM for {pred_len}h horizon...")

    from utils.logger import get_device
    device = get_device()

    # Load training windows from train set
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv", index_col="timestamp", parse_dates=True)
    train_values = train_df["ALLSKY_SFC_SW_DWN"].interpolate(limit=3).dropna().values

    stride = 24
    train_contexts, train_targets = [], []
    for i in range(0, len(train_values) - CONTEXT_LENGTH - pred_len + 1, stride):
        ctx = train_values[i : i + CONTEXT_LENGTH]
        tgt = train_values[i + CONTEXT_LENGTH : i + CONTEXT_LENGTH + pred_len]
        if not (np.isnan(ctx).any() or np.isnan(tgt).any()):
            train_contexts.append(ctx)
            train_targets.append(tgt)

    # Add validation windows to training
    val_ctx, val_tgt, _ = load_windows("val", pred_len)
    train_contexts = np.array(train_contexts + list(val_ctx))
    train_targets = np.array(train_targets + list(val_tgt))

    # Normalize
    scaler = StandardScaler()
    train_contexts_flat = scaler.fit_transform(train_contexts.reshape(-1, 1)).reshape(train_contexts.shape)

    # Load test
    test_contexts, test_targets, test_timestamps = load_windows("test", pred_len)
    test_contexts_flat = scaler.transform(test_contexts.reshape(-1, 1)).reshape(test_contexts.shape)

    # Scale targets too
    target_scaler = StandardScaler()
    train_targets_scaled = target_scaler.fit_transform(train_targets)

    # PyTorch datasets
    X_train = torch.tensor(train_contexts_flat, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(train_targets_scaled, dtype=torch.float32)
    X_test = torch.tensor(test_contexts_flat, dtype=torch.float32).unsqueeze(-1)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # Model
    model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2, output_size=pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Train with early stopping
    max_epochs = 200
    patience = 20
    best_loss = float("inf")
    epochs_no_improve = 0

    model.train()
    for epoch in range(max_epochs):
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}/{max_epochs}, Loss: {avg_loss:.4f}")

        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch + 1} (best loss: {best_loss:.4f})")
            break

    model.load_state_dict(best_state)

    # Predict
    model.eval()
    with torch.no_grad():
        X_test_dev = X_test.to(device)
        preds_scaled = model(X_test_dev).cpu().numpy()

    predictions = target_scaler.inverse_transform(preds_scaled)
    predictions = np.clip(predictions, 0, None)
    return predictions, test_targets, test_timestamps


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = create_log("eval_baselines", {
        "models": ["XGBoost", "LSTM"],
        "prediction_lengths": PREDICTION_LENGTHS,
        "context_length": CONTEXT_LENGTH,
    })
    all_results = []

    for pred_len in PREDICTION_LENGTHS:
        print(f"\n{'=' * 60}")
        print(f"Prediction horizon: {pred_len}h")
        print(f"{'=' * 60}")

        _, targets_ref, timestamps_ref = load_windows("test", pred_len)
        timestamps = pd.DatetimeIndex(timestamps_ref)
        seasons = np.array([get_season(m) for m in timestamps.month])

        # XGBoost
        log_step(log, f"xgboost_{pred_len}h", {"horizon": pred_len})
        xgb_preds, xgb_targets, _ = train_and_eval_xgboost(pred_len)
        for season_label in ["all", "dry", "wet"]:
            mask = np.ones(len(xgb_targets), dtype=bool) if season_label == "all" else (seasons == season_label)
            m = compute_metrics(xgb_preds[mask], xgb_targets[mask])
            all_results.append({"model": "XGBoost", "horizon": f"{pred_len}h", "season": season_label, **m})
        xgb_overall = compute_metrics(xgb_preds, xgb_targets)
        print(f"  XGBoost MAE: {xgb_overall['MAE']:.2f}")
        log_step(log, f"xgboost_{pred_len}h_done", {"MAE": xgb_overall["MAE"]})

        np.save(RESULTS_DIR.parent / f"preds_xgboost_{pred_len}h.npy", xgb_preds)

        # LSTM
        log_step(log, f"lstm_{pred_len}h", {"horizon": pred_len})
        lstm_preds, lstm_targets, _ = train_and_eval_lstm(pred_len)
        for season_label in ["all", "dry", "wet"]:
            mask = np.ones(len(lstm_targets), dtype=bool) if season_label == "all" else (seasons == season_label)
            m = compute_metrics(lstm_preds[mask], lstm_targets[mask])
            all_results.append({"model": "LSTM", "horizon": f"{pred_len}h", "season": season_label, **m})
        lstm_overall = compute_metrics(lstm_preds, lstm_targets)
        print(f"  LSTM MAE: {lstm_overall['MAE']:.2f}")
        log_step(log, f"lstm_{pred_len}h_done", {"MAE": lstm_overall["MAE"]})

        np.save(RESULTS_DIR.parent / f"preds_lstm_{pred_len}h.npy", lstm_preds)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)
    save_summary(all_results, "baseline_results")
    save_log(log)
    print(f"\nResults saved to results/tables/baseline_results.csv")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
