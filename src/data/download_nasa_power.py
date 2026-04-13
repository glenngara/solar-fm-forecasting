"""Download hourly solar irradiance and weather data from NASA POWER API for Laguna de Bay."""

import sys
import json
import requests
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR

API_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
LAT = 14.3833
LON = 121.2500
PARAMETERS = "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,RH2M,WS2M,CLOUD_AMT"
OUTPUT_DIR = RAW_DIR


def download_year(year: int) -> dict:
    """Download one year of hourly data from NASA POWER."""
    params = {
        "parameters": PARAMETERS,
        "community": "RE",
        "longitude": LON,
        "latitude": LAT,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON",
    }
    print(f"Downloading {year}...")
    resp = requests.get(API_URL, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def parse_response(data: dict) -> pd.DataFrame:
    """Parse NASA POWER JSON response into a DataFrame."""
    params = data["properties"]["parameter"]
    records = {}
    for param_name, hourly_values in params.items():
        for timestamp, value in hourly_values.items():
            records.setdefault(timestamp, {})[param_name] = value

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "timestamp"
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    df = df.sort_index()
    # Replace fill values (-999) with NaN
    df = df.replace(-999.0, pd.NA)
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "nasa_power_laguna_de_bay_2020_2025.csv"

    if out_path.exists():
        print(f"Data already exists: {out_path}")
        print("Skipping download. Delete the file to re-download.")
        return

    frames = []

    for year in range(2020, 2025 + 1):
        data = download_year(year)
        df = parse_response(data)
        frames.append(df)
        print(f"  {year}: {len(df)} records")

    combined = pd.concat(frames)
    out_path = OUTPUT_DIR / "nasa_power_laguna_de_bay_2020_2025.csv"
    combined.to_csv(out_path)
    print(f"\nSaved {len(combined)} records to {out_path}")

    # Quick summary
    print("\n--- Data Summary ---")
    print(combined.describe().round(2))
    print(f"\nDate range: {combined.index.min()} to {combined.index.max()}")
    print(f"Missing values:\n{combined.isna().sum()}")


if __name__ == "__main__":
    main()
