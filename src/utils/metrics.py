"""Shared evaluation metrics for solar irradiance forecasting.

Provides:
- MAE, RMSE (standard point metrics)
- MASE (scale-free, comparable across datasets)
- CRPS (probabilistic calibration — gold standard for probabilistic forecasts)
- Diebold-Mariano test (statistical significance between model pairs)
"""

import numpy as np
from scipy import stats


def mae(predictions, targets):
    """Mean Absolute Error."""
    return np.mean(np.abs(predictions - targets))


def rmse(predictions, targets):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((predictions - targets) ** 2))


def mase(predictions, targets, seasonal_period=24):
    """Mean Absolute Scaled Error.

    Normalizes MAE by the naive seasonal forecast error.
    MASE < 1 means the model beats the seasonal naive baseline.

    Args:
        predictions: (n_windows, pred_len) predicted values
        targets: (n_windows, pred_len) true values
        seasonal_period: period for the naive forecast (24 = daily cycle for hourly data)
    """
    forecast_errors = np.mean(np.abs(predictions - targets))

    # Naive seasonal error: use the seasonal differencing of the target
    # If prediction length <= seasonal_period, we can't compute within-target differencing,
    # so we use the mean absolute difference between consecutive values as fallback
    if targets.shape[1] > seasonal_period:
        naive_errors = np.mean(np.abs(targets[:, seasonal_period:] - targets[:, :-seasonal_period]))
    else:
        # Fallback: use step-wise differencing as naive error
        naive_errors = np.mean(np.abs(np.diff(targets, axis=1)))

    if naive_errors == 0 or np.isnan(naive_errors):
        return np.inf
    return forecast_errors / naive_errors


def crps_empirical(samples, targets):
    """Continuous Ranked Probability Score (empirical/ensemble version).

    Measures how well probabilistic forecasts are calibrated.
    Lower is better. Uses the energy form of CRPS.

    Args:
        samples: (n_windows, n_samples, pred_len) forecast samples
        targets: (n_windows, pred_len) true values

    Returns:
        Mean CRPS across all windows and time steps.
    """
    # samples: (N, S, T), targets: (N, T)
    n_windows, n_samples, pred_len = samples.shape

    # Term 1: E[|X - y|]
    abs_diff = np.abs(samples - targets[:, np.newaxis, :])  # (N, S, T)
    term1 = np.mean(abs_diff, axis=1)  # (N, T)

    # Term 2: 0.5 * E[|X - X'|]
    # For efficiency, use the sorted samples trick
    sorted_samples = np.sort(samples, axis=1)  # (N, S, T)
    # E[|X - X'|] = 2 * sum_{i=1}^{S} (2i - S - 1) * x_{(i)} / (S^2)
    weights = 2 * np.arange(1, n_samples + 1) - n_samples - 1  # (S,)
    weights = weights[np.newaxis, :, np.newaxis]  # (1, S, 1)
    term2 = np.sum(weights * sorted_samples, axis=1) / (n_samples ** 2)  # (N, T)

    crps = term1 - term2  # (N, T)
    return np.mean(crps)


def compute_all_metrics(predictions, targets, samples=None, seasonal_period=24):
    """Compute all point metrics, plus CRPS if samples are provided.

    Args:
        predictions: (n_windows, pred_len) point predictions (e.g., median)
        targets: (n_windows, pred_len) true values
        samples: optional (n_windows, n_samples, pred_len) for CRPS
        seasonal_period: for MASE computation

    Returns:
        Dict with MAE, RMSE, MASE, and optionally CRPS.
    """
    result = {
        "MAE": mae(predictions, targets),
        "RMSE": rmse(predictions, targets),
        "MASE": mase(predictions, targets, seasonal_period),
    }
    if samples is not None:
        result["CRPS"] = crps_empirical(samples, targets)
    return result


def diebold_mariano_test(errors_1, errors_2, horizon=1):
    """Diebold-Mariano test for equal predictive accuracy.

    Tests H0: both models have equal forecast accuracy.

    Args:
        errors_1: (n_windows, pred_len) absolute errors from model 1
        errors_2: (n_windows, pred_len) absolute errors from model 2
        horizon: forecast horizon for Newey-West bandwidth

    Returns:
        Dict with DM statistic and p-value.
        p < 0.05 means the difference is statistically significant.
    """
    # Loss differential (using squared errors)
    d = errors_1 ** 2 - errors_2 ** 2

    # Average across prediction length
    d_mean = np.mean(d, axis=1)  # (n_windows,)

    n = len(d_mean)
    d_bar = np.mean(d_mean)

    # Newey-West variance estimator
    gamma_0 = np.mean((d_mean - d_bar) ** 2)
    bandwidth = horizon - 1

    gamma_sum = 0
    for k in range(1, bandwidth + 1):
        gamma_k = np.mean((d_mean[k:] - d_bar) * (d_mean[:-k] - d_bar))
        gamma_sum += 2 * gamma_k

    variance = (gamma_0 + gamma_sum) / n

    if variance <= 0:
        return {"DM_statistic": 0.0, "p_value": 1.0}

    dm_stat = d_bar / np.sqrt(variance)
    p_value = 2 * stats.norm.sf(np.abs(dm_stat))

    return {"DM_statistic": dm_stat, "p_value": p_value}


def pairwise_dm_tests(model_errors, model_names, horizon=1):
    """Run DM tests for all model pairs.

    Args:
        model_errors: dict of {model_name: (n_windows, pred_len) absolute errors}
        model_names: list of model names to compare
        horizon: forecast horizon

    Returns:
        List of dicts with model pairs and test results.
    """
    results = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a = model_names[i]
            name_b = model_names[j]
            dm = diebold_mariano_test(
                model_errors[name_a], model_errors[name_b], horizon
            )
            sig = "***" if dm["p_value"] < 0.001 else "**" if dm["p_value"] < 0.01 else "*" if dm["p_value"] < 0.05 else "ns"
            results.append({
                "model_A": name_a,
                "model_B": name_b,
                **dm,
                "significance": sig,
            })
    return results
