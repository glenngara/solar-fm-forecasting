"""Generate all paper figures from evaluation results.

Figures:
1. Overall model comparison bar chart (MAE by horizon)
2. Zero-shot vs fine-tuned improvement chart
3. Seasonal performance heatmap (dry vs wet)
4. Ablation study: MAE vs training steps
5. CRPS comparison (probabilistic calibration)
6. Forecast examples: best/worst predictions
7. Diebold-Mariano significance matrix
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, TABLES_DIR, FIGURES_DIR, PROCESSED_DIR

# Consistent style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

COLORS = {
    "Persistence": "#999999",
    "XGBoost": "#2ca02c",
    "LSTM": "#d62728",
    "Chronos-2 (ZS)": "#1f77b4",
    "Chronos-T5-Small (ZS)": "#aec7e8",
    "Chronos-T5-Small (FT)": "#6baed6",
    "Chronos-T5-Base (ZS)": "#ff7f0e",
    "Chronos-T5-Base (FT)": "#ffbb78",
    "TimesFM-2.5 (ZS)": "#9467bd",
    "Moirai-2.0-Small (ZS)": "#8c564b",
    "TTM-R2 (ZS)": "#e377c2",
    "TTM-R2 (FT)": "#f7b6d2",
}


def load_results():
    """Load all available result CSVs."""
    results = {}
    for csv_file in TABLES_DIR.glob("*.csv"):
        results[csv_file.stem] = pd.read_csv(csv_file)
    return results


def fig_model_comparison_bar(results):
    """Fig: Overall MAE comparison grouped by horizon."""
    # Try all_models_comparison first, fall back to individual files
    if "all_models_comparison" in results:
        df = results["all_models_comparison"]
    else:
        dfs = []
        for key in ["zero_shot_results", "baseline_results", "finetuned_comparison"]:
            if key in results:
                dfs.append(results[key])
        if not dfs:
            print("  No results found for model comparison")
            return
        df = pd.concat(dfs, ignore_index=True)

    overall = df[df["season"] == "all"].copy()

    for pred_len in ["24h", "72h"]:
        subset = overall[overall["horizon"] == pred_len].sort_values("MAE")
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = [COLORS.get(m, "#666666") for m in subset["model"]]
        bars = ax.barh(range(len(subset)), subset["MAE"], color=colors, edgecolor="white")

        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels(subset["model"])
        ax.set_xlabel("MAE (W/m²)")
        ax.set_title(f"Model Comparison — {pred_len} Forecast Horizon")
        ax.invert_yaxis()

        # Add value labels
        for bar, val in zip(bars, subset["MAE"]):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"fig_model_comparison_{pred_len}.png", bbox_inches="tight")
        plt.close()
        print(f"  Saved fig_model_comparison_{pred_len}.png")


def fig_zeroshot_vs_finetuned(results):
    """Fig: Side-by-side zero-shot vs fine-tuned for each model."""
    if "finetuned_comparison" in results:
        df = results["finetuned_comparison"]
    elif "all_models_comparison" in results:
        df = results["all_models_comparison"]
        df = df[df["model"].str.contains("ZS|FT|zero-shot|fine-tuned", regex=True)]
    else:
        print("  No fine-tuned results found")
        return

    overall = df[df["season"] == "all"].copy()

    for pred_len in ["24h", "72h"]:
        subset = overall[overall["horizon"] == pred_len]
        if subset.empty:
            continue

        # Group by model family
        fig, ax = plt.subplots(figsize=(10, 6))

        models_zs = subset[subset["model"].str.contains("ZS|zero-shot", regex=True)]
        models_ft = subset[subset["model"].str.contains("FT|fine-tuned", regex=True)]

        if models_zs.empty or models_ft.empty:
            continue

        x = np.arange(len(models_zs))
        width = 0.35

        zs_labels = [m.split("(")[0].strip() for m in models_zs["model"]]

        bars1 = ax.bar(x - width / 2, models_zs["MAE"].values, width,
                       label="Zero-shot", color="#1f77b4", edgecolor="white")
        bars2 = ax.bar(x + width / 2, models_ft["MAE"].values, width,
                       label="Fine-tuned", color="#ff7f0e", edgecolor="white")

        ax.set_ylabel("MAE (W/m²)")
        ax.set_title(f"Zero-Shot vs Fine-Tuned — {pred_len} Horizon")
        ax.set_xticks(x)
        ax.set_xticklabels(zs_labels)
        ax.legend()

        # Add improvement labels
        for i, (zs_val, ft_val) in enumerate(zip(models_zs["MAE"].values, models_ft["MAE"].values)):
            imp = (1 - ft_val / zs_val) * 100
            color = "green" if imp > 0 else "red"
            ax.text(i, max(zs_val, ft_val) + 0.5, f"{imp:+.1f}%",
                    ha="center", fontsize=10, fontweight="bold", color=color)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"fig_zs_vs_ft_{pred_len}.png", bbox_inches="tight")
        plt.close()
        print(f"  Saved fig_zs_vs_ft_{pred_len}.png")


def fig_seasonal_heatmap(results):
    """Fig: Seasonal performance heatmap (dry vs wet)."""
    if "all_models_comparison" in results:
        df = results["all_models_comparison"]
    elif "finetuned_comparison" in results:
        df = results["finetuned_comparison"]
    else:
        print("  No results for seasonal heatmap")
        return

    for pred_len in ["24h", "72h"]:
        subset = df[df["horizon"] == pred_len]
        if subset.empty:
            continue

        pivot = subset.pivot_table(index="model", columns="season", values="MAE")
        if "all" in pivot.columns:
            pivot = pivot[["dry", "wet", "all"]]

        fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.5)))
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([c.title() for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        # Add text
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                color = "white" if val > pivot.values.mean() else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        color=color, fontsize=10)

        ax.set_title(f"MAE by Season — {pred_len} Horizon")
        plt.colorbar(im, ax=ax, label="MAE (W/m²)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"fig_seasonal_heatmap_{pred_len}.png", bbox_inches="tight")
        plt.close()
        print(f"  Saved fig_seasonal_heatmap_{pred_len}.png")


def fig_ablation_steps(results):
    """Fig: MAE vs training steps (ablation study)."""
    if "ablation_steps" not in results:
        print("  No ablation results found (run eval_ablation.py first)")
        return

    df = results["ablation_steps"]
    overall = df[df["season"] == "all"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, pred_len in enumerate(["24h", "72h"]):
        ax = axes[idx]
        subset = overall[overall["horizon"] == pred_len].copy()
        if subset.empty:
            continue

        # Parse step numbers for x-axis
        def parse_steps(s):
            s = str(s)
            if "zero" in s.lower():
                return 0
            if "final" in s.lower():
                return 5000
            try:
                return int(s)
            except ValueError:
                return -1
        subset["step_num"] = subset["steps"].apply(parse_steps)
        subset = subset[subset["step_num"] >= 0].sort_values("step_num")

        ax.plot(subset["step_num"], subset["MAE"], "o-", color="#1f77b4",
                linewidth=2, markersize=8, label="MAE")
        ax.plot(subset["step_num"], subset["RMSE"], "s--", color="#ff7f0e",
                linewidth=2, markersize=8, label="RMSE")

        ax.axhline(y=subset[subset["step_num"] == 0]["MAE"].values[0],
                    color="#1f77b4", linestyle=":", alpha=0.5, label="Zero-shot MAE")

        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Error (W/m²)")
        ax.set_title(f"Ablation: Training Steps — {pred_len} Horizon")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_ablation_steps.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig_ablation_steps.png")


def fig_crps_comparison(results):
    """Fig: CRPS comparison for probabilistic models."""
    if "all_models_comparison" in results:
        df = results["all_models_comparison"]
    elif "finetuned_comparison" in results:
        df = results["finetuned_comparison"]
    else:
        print("  No CRPS results found")
        return

    if "CRPS" not in df.columns:
        print("  No CRPS column in results")
        return

    overall = df[(df["season"] == "all") & (df["CRPS"].notna())].copy()

    for pred_len in ["24h", "72h"]:
        subset = overall[overall["horizon"] == pred_len].sort_values("CRPS")
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [COLORS.get(m, "#666666") for m in subset["model"]]
        bars = ax.barh(range(len(subset)), subset["CRPS"], color=colors, edgecolor="white")

        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels(subset["model"])
        ax.set_xlabel("CRPS (lower is better)")
        ax.set_title(f"Probabilistic Calibration (CRPS) — {pred_len} Horizon")
        ax.invert_yaxis()

        for bar, val in zip(bars, subset["CRPS"]):
            ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"fig_crps_{pred_len}.png", bbox_inches="tight")
        plt.close()
        print(f"  Saved fig_crps_{pred_len}.png")


def fig_forecast_examples(results):
    """Fig: Example forecast plots showing best and worst predictions."""
    for pred_len in [24, 72]:
        # Load test data
        data = np.load(PROCESSED_DIR / f"test_windows_{pred_len}h.npz", allow_pickle=True)
        contexts = data["contexts"]
        targets = data["targets"]

        # Find prediction files
        pred_files = list(RESULTS_DIR.glob(f"preds_*_{pred_len}h.npy"))
        if not pred_files:
            continue

        # Pick best model (lowest MAE from results)
        best_model = None
        best_preds = None
        best_mae = float("inf")
        for pf in pred_files:
            preds = np.load(pf)
            if preds.shape == targets.shape:
                mae = np.mean(np.abs(preds - targets))
                if mae < best_mae:
                    best_mae = mae
                    best_preds = preds
                    best_model = pf.stem.replace(f"preds_", "").replace(f"_{pred_len}h", "")

        if best_preds is None:
            continue

        # Find best and worst windows
        window_maes = np.mean(np.abs(best_preds - targets), axis=1)
        best_idx = np.argmin(window_maes)
        worst_idx = np.argmax(window_maes)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        for ax, idx, label in [(axes[0], best_idx, "Best"), (axes[1], worst_idx, "Worst")]:
            ctx = contexts[idx]
            tgt = targets[idx]
            pred = best_preds[idx]

            t_ctx = range(len(ctx))
            t_pred = range(len(ctx), len(ctx) + len(tgt))

            ax.plot(t_ctx, ctx, color="#1f77b4", linewidth=1.5, label="Context")
            ax.plot(t_pred, tgt, color="#2ca02c", linewidth=2, label="Actual")
            ax.plot(t_pred, pred, color="#ff7f0e", linewidth=2, linestyle="--", label="Forecast")
            ax.axvline(x=len(ctx), color="gray", linestyle=":", alpha=0.5)
            ax.set_title(f"{label} Forecast (MAE={window_maes[idx]:.1f})")
            ax.set_xlabel("Hours")
            ax.set_ylabel("Solar Irradiance (W/m²)")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Forecast Examples — {pred_len}h Horizon ({best_model})", fontsize=14)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"fig_forecast_examples_{pred_len}h.png", bbox_inches="tight")
        plt.close()
        print(f"  Saved fig_forecast_examples_{pred_len}h.png")


def fig_dm_significance(results):
    """Fig: Diebold-Mariano test significance matrix."""
    for pred_len in ["24h", "72h"]:
        dm_key = f"dm_tests_all_{pred_len}"
        if dm_key not in results:
            dm_key = f"dm_tests_{pred_len}"
        if dm_key not in results:
            continue

        dm_df = results[dm_key]
        models = sorted(set(dm_df["model_A"]) | set(dm_df["model_B"]))

        # Build matrix
        n = len(models)
        matrix = np.ones((n, n))  # 1 = not significant
        for _, row in dm_df.iterrows():
            i = models.index(row["model_A"])
            j = models.index(row["model_B"])
            matrix[i, j] = row["p_value"]
            matrix[j, i] = row["p_value"]

        fig, ax = plt.subplots(figsize=(max(8, n), max(6, n * 0.8)))
        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=0.1, aspect="auto")

        ax.set_xticks(range(n))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(models, fontsize=9)

        for i in range(n):
            for j in range(n):
                if i != j:
                    p = matrix[i, j]
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    color = "white" if p < 0.05 else "black"
                    ax.text(j, i, sig, ha="center", va="center", color=color, fontsize=9)

        ax.set_title(f"Statistical Significance (DM Test) — {pred_len}")
        plt.colorbar(im, ax=ax, label="p-value")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"fig_dm_significance_{pred_len}.png", bbox_inches="tight")
        plt.close()
        print(f"  Saved fig_dm_significance_{pred_len}.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()

    print(f"Available result files: {list(results.keys())}")
    print(f"\nGenerating figures...")

    fig_model_comparison_bar(results)
    fig_zeroshot_vs_finetuned(results)
    fig_seasonal_heatmap(results)
    fig_ablation_steps(results)
    fig_crps_comparison(results)
    fig_forecast_examples(results)
    fig_dm_significance(results)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
