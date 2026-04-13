"""Exploratory Data Analysis for Laguna de Bay solar irradiance.

Generates figures for understanding seasonal patterns, wet vs dry season contrast,
and data characteristics relevant to the paper.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, FIGURES_DIR, RAW_FILENAME

DATA_PATH = RAW_DIR / RAW_FILENAME
FIG_DIR = FIGURES_DIR

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def load_data():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    df = df.replace(-999.0, np.nan)
    return df


def fig1_annual_irradiance_profile(df):
    """Monthly mean solar irradiance across all years — dry vs wet season."""
    monthly = df.groupby(df.index.month)["ALLSKY_SFC_SW_DWN"].agg(["mean", "std"])
    months = range(1, 13)
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#2196F3" if m in [6, 7, 8, 9, 10, 11] else "#FF9800" for m in months]
    bars = ax.bar(months, monthly["mean"], yerr=monthly["std"], capsize=3,
                  color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(list(months))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Solar Irradiance (W/m²)")
    ax.set_title("Mean Hourly Solar Irradiance at Laguna de Bay (2020–2025)")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#FF9800", label="Dry Season (Dec–May)"),
        Patch(facecolor="#2196F3", label="Wet/Monsoon Season (Jun–Nov)"),
    ], loc="upper right")

    ax.set_ylim(0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_annual_irradiance_profile.png")
    plt.close()
    print("Saved fig1_annual_irradiance_profile.png")


def fig2_daily_irradiance_heatmap(df):
    """Heatmap of hourly irradiance across months — shows diurnal + seasonal patterns."""
    # Use 2025 test year
    test = df[df.index.year == 2025].copy()
    test["hour"] = test.index.hour
    test["month"] = test.index.month

    pivot = test.pivot_table(
        values="ALLSKY_SFC_SW_DWN", index="hour", columns="month", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", origin="lower",
                   extent=[0.5, 12.5, -0.5, 23.5])
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Hour of Day")
    ax.set_xlabel("Month")
    ax.set_title("Hourly Solar Irradiance Pattern — Laguna de Bay (2025)")
    cbar = fig.colorbar(im, ax=ax, label="Irradiance (W/m²)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_daily_irradiance_heatmap.png")
    plt.close()
    print("Saved fig2_daily_irradiance_heatmap.png")


def fig3_wet_vs_dry_timeseries(df):
    """Side-by-side: one week in dry season vs one week in wet season."""
    dry_week = df["2025-03-01":"2025-03-07"]["ALLSKY_SFC_SW_DWN"]
    wet_week = df["2025-08-01":"2025-08-07"]["ALLSKY_SFC_SW_DWN"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    ax1.plot(range(len(dry_week)), dry_week.values, color="#FF9800", linewidth=0.8)
    ax1.fill_between(range(len(dry_week)), dry_week.values, alpha=0.3, color="#FF9800")
    ax1.set_title("Dry Season — March 1–7, 2025")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Solar Irradiance (W/m²)")
    ax1.set_ylim(0)
    ax1.grid(alpha=0.3)

    ax2.plot(range(len(wet_week)), wet_week.values, color="#2196F3", linewidth=0.8)
    ax2.fill_between(range(len(wet_week)), wet_week.values, alpha=0.3, color="#2196F3")
    ax2.set_title("Wet/Monsoon Season — August 1–7, 2025")
    ax2.set_xlabel("Hour")
    ax2.set_ylim(0)
    ax2.grid(alpha=0.3)

    fig.suptitle("Solar Irradiance: Dry vs Wet Season at Laguna de Bay", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_wet_vs_dry_timeseries.png", bbox_inches="tight")
    plt.close()
    print("Saved fig3_wet_vs_dry_timeseries.png")


def fig4_cloud_cover_vs_irradiance(df):
    """Scatter: cloud cover vs irradiance, colored by season."""
    daytime = df[(df.index.hour >= 6) & (df.index.hour <= 18)].copy()
    daytime = daytime.dropna(subset=["CLOUD_AMT", "ALLSKY_SFC_SW_DWN"])
    daytime["season"] = daytime.index.month.map(
        lambda m: "Dry" if m in [12, 1, 2, 3, 4, 5] else "Wet"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for season, color in [("Dry", "#FF9800"), ("Wet", "#2196F3")]:
        subset = daytime[daytime["season"] == season]
        ax.scatter(subset["CLOUD_AMT"], subset["ALLSKY_SFC_SW_DWN"],
                   alpha=0.05, s=5, color=color, label=season)

    ax.set_xlabel("Cloud Amount (%)")
    ax.set_ylabel("Solar Irradiance (W/m²)")
    ax.set_title("Cloud Cover vs Solar Irradiance (Daytime Only)")
    ax.legend(markerscale=5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_cloud_vs_irradiance.png")
    plt.close()
    print("Saved fig4_cloud_vs_irradiance.png")


def fig5_yearly_comparison(df):
    """Daily mean irradiance for each year overlaid — shows inter-annual variability."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 6))

    for i, year in enumerate(range(2020, 2026)):
        yearly = df[df.index.year == year]["ALLSKY_SFC_SW_DWN"]
        daily_mean = yearly.resample("D").mean()
        # Smooth with 7-day rolling average
        smoothed = daily_mean.rolling(7, center=True).mean()
        day_of_year = smoothed.index.dayofyear
        ax.plot(day_of_year, smoothed.values, color=colors[i], label=str(year),
                linewidth=1.2, alpha=0.8)

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Daily Mean Irradiance (W/m²)")
    ax.set_title("Inter-Annual Solar Irradiance Variability — Laguna de Bay")
    ax.legend(ncol=6, loc="upper right")
    ax.set_xlim(1, 365)
    ax.grid(alpha=0.3)

    # Mark monsoon season
    ax.axvspan(152, 335, alpha=0.08, color="blue", label="Monsoon")
    ax.text(243, ax.get_ylim()[1] * 0.95, "Monsoon Season", ha="center",
            fontsize=10, color="#2196F3", style="italic")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_yearly_comparison.png")
    plt.close()
    print("Saved fig5_yearly_comparison.png")


def print_summary_stats(df):
    """Print key statistics for the paper."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    daytime = df[(df.index.hour >= 6) & (df.index.hour <= 18)]

    print(f"\nTotal records: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Missing ALLSKY_SFC_SW_DWN: {df['ALLSKY_SFC_SW_DWN'].isna().sum()}")

    print(f"\n--- Daytime Irradiance (6AM-6PM) ---")
    print(f"Mean: {daytime['ALLSKY_SFC_SW_DWN'].mean():.1f} W/m²")
    print(f"Std:  {daytime['ALLSKY_SFC_SW_DWN'].std():.1f} W/m²")
    print(f"Max:  {daytime['ALLSKY_SFC_SW_DWN'].max():.1f} W/m²")

    # Seasonal comparison
    dry = daytime[daytime.index.month.isin([12, 1, 2, 3, 4, 5])]
    wet = daytime[daytime.index.month.isin([6, 7, 8, 9, 10, 11])]
    print(f"\n--- Seasonal Comparison (Daytime) ---")
    print(f"Dry season mean:  {dry['ALLSKY_SFC_SW_DWN'].mean():.1f} W/m²")
    print(f"Wet season mean:  {wet['ALLSKY_SFC_SW_DWN'].mean():.1f} W/m²")
    print(f"Reduction:        {(1 - wet['ALLSKY_SFC_SW_DWN'].mean() / dry['ALLSKY_SFC_SW_DWN'].mean()) * 100:.1f}%")

    print(f"\n--- Climate ---")
    print(f"Mean temperature: {df['T2M'].mean():.1f}°C")
    print(f"Mean humidity:    {df['RH2M'].mean():.1f}%")
    print(f"Mean cloud cover: {df['CLOUD_AMT'].dropna().mean():.1f}%")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    print("Generating EDA figures...\n")
    fig1_annual_irradiance_profile(df)
    fig2_daily_irradiance_heatmap(df)
    fig3_wet_vs_dry_timeseries(df)
    fig4_cloud_cover_vs_irradiance(df)
    fig5_yearly_comparison(df)
    print_summary_stats(df)


if __name__ == "__main__":
    main()
