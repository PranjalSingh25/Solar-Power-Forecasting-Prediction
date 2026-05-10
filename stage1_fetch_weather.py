"""
Stage 1 — Weather Data Fetcher
Fetches hourly weather data from NASA POWER API for any location.
Handles multi-year requests by chunking into annual batches.
"""

import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR  = BASE_DIR / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

RAW_CSV = DATA_DIR / "nasa_power_hourly_raw.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage1_fetch.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("stage1")

NASA_PARAMS = ["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF",
               "T2M", "WS10M", "RH2M"]
API_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"


def _make_session():
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def fetch_one_year(session, lat: float, lon: float, year: int) -> pd.DataFrame | None:
    """Fetch a single calendar year of hourly data."""
    start = f"{year}0101"
    end   = f"{year}1231"
    params = {
        "parameters": ",".join(NASA_PARAMS),
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": "LST",
    }
    log.info(f"  Fetching {year} for ({lat}, {lon}) …")
    try:
        r = session.get(API_URL, params=params, timeout=90)
        r.raise_for_status()
        raw = r.json()
        pdata = raw.get("properties", {}).get("parameter", {})
        if not pdata:
            log.warning(f"  No data returned for {year}")
            return None

        dfs = []
        for p in NASA_PARAMS:
            if p in pdata:
                vals = {k: (np.nan if v == -999 else v)
                        for k, v in pdata[p].items()}
                dfs.append(pd.DataFrame.from_dict(vals, orient="index", columns=[p]))
        if not dfs:
            return None

        df = pd.concat(dfs, axis=1)
        df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
        df.index.name = "Timestamp"
        df.sort_index(inplace=True)
        log.info(f"  ✓ {year}: {len(df)} hourly rows")
        return df

    except Exception as e:
        log.error(f"  ✗ Failed {year}: {e}")
        return None


def fetch_weather(lat: float, lon: float,
                  start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetch multiple years and concatenate.
    Returns a combined DataFrame saved to data/nasa_power_hourly_raw.csv
    """
    session = _make_session()
    frames = []
    for yr in range(start_year, end_year + 1):
        df = fetch_one_year(session, lat, lon, yr)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError("No data fetched — check coordinates and date range.")

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    # Fill isolated NaNs via linear interpolation (max 3-hour gaps)
    combined = combined.interpolate(method="time", limit=3)
    combined.dropna(inplace=True)

    combined.to_csv(RAW_CSV)
    log.info(f"Saved {len(combined)} rows → {RAW_CSV}")
    return combined


if __name__ == "__main__":
    import sys
    print("=== Stage 1: NASA POWER Weather Fetch ===")
    try:
        lat   = float(input("Latitude  (-90 to 90):  ").strip() or "28.6139")
        lon   = float(input("Longitude (-180 to 180): ").strip() or "77.2090")
        sy    = int(input("Start year (e.g. 2018):  ").strip() or "2018")
        ey    = int(input("End year   (e.g. 2023):  ").strip() or "2023")
    except ValueError:
        print("Invalid input. Using defaults: New Delhi 2018-2023")
        lat, lon, sy, ey = 28.6139, 77.2090, 2018, 2023

    df = fetch_weather(lat, lon, sy, ey)
    print(f"\n✓ Fetched {len(df):,} hourly rows ({sy}–{ey})")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Saved to: {RAW_CSV}")
