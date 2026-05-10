"""
Stage 2 — PV Physics Simulation + Shadow Analysis
Runs pvlib ModelChain on hourly weather data.
Also computes a horizon-based shading loss factor per hour.
"""

import pvlib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from timezonefinder import TimezoneFinder

BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
LOG_DIR     = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

INPUT_CSV   = DATA_DIR / "nasa_power_hourly_raw.csv"
OUTPUT_CSV  = DATA_DIR / "processed" / "weather_and_simulated_hourly_power.csv"
OUTPUT_CSV.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage2_simulate.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("stage2")

# ─── PV System Configuration ──────────────────────────────────────────────────
# System calibrated for the existing simulation data (New Delhi, 3 kWp system)
# Change these to match any real system.

LATITUDE          = 28.6139
LONGITUDE         = 77.2090
ALTITUDE          = 216          # metres
SURFACE_TILT      = 28           # degrees from horizontal
SURFACE_AZIMUTH   = 180          # 180 = south-facing

MODULE_NAME       = "Canadian_Solar_Inc__CS6X_300M"
INVERTER_NAME     = "SMA_America__SB7000TL_US__240V_"
MODULES_PER_STRING = 10
STRINGS_PER_INV   = 2            # 10×300W × 2 strings = 6 kWp

SYSTEM_LOSSES     = 0.14         # 14% total system losses (wiring, dust, mismatch)

# Horizon shading: define as list of (azimuth_deg, horizon_elevation_deg) pairs.
# For flat rooftops in urban areas 5–10° is a reasonable default.
# More precise values can be obtained from a site survey.
HORIZON_AZIMUTHS    = list(range(0, 360, 10))
HORIZON_ELEVATIONS  = [5.0] * 36   # 5° obstruction all around (conservative urban estimate)


def compute_shading_loss(times: pd.DatetimeIndex,
                          lat: float, lon: float) -> pd.Series:
    """
    Returns an hourly shading factor (0–1) where 1 = no shading.
    Uses pvlib sun position + a simple horizon profile.
    If sun elevation < local horizon elevation at that azimuth → shaded (0).
    """
    loc = pvlib.location.Location(lat, lon)
    solpos = loc.get_solarposition(times)

    horizon_series = pd.Series(
        np.interp(
            solpos["azimuth"],
            HORIZON_AZIMUTHS + [360],
            HORIZON_ELEVATIONS + [HORIZON_ELEVATIONS[0]],
        ),
        index=times,
    )

    # 1 = clear, 0 = shaded by horizon
    shading_factor = (solpos["elevation"] > horizon_series).astype(float)
    shading_factor[solpos["elevation"] <= 0] = 0.0  # nighttime always 0
    return shading_factor


def simulate(weather_csv: Path = INPUT_CSV,
             output_csv: Path  = OUTPUT_CSV) -> pd.DataFrame:

    log.info(f"Loading weather data from {weather_csv}")
    df = pd.read_csv(weather_csv, index_col="Timestamp", parse_dates=True)
    log.info(f"  {len(df):,} hourly rows loaded")

    # ── Timezone ──────────────────────────────────────────────────────────────
    tf = TimezoneFinder()
    tz = tf.timezone_at(lng=LONGITUDE, lat=LATITUDE) or "Asia/Kolkata"
    log.info(f"  Timezone: {tz}")

    if df.index.tz is None:
        df = df.tz_localize(tz)
    else:
        df = df.tz_convert(tz)

    # ── Column rename for pvlib ───────────────────────────────────────────────
    rename = {
        "ALLSKY_SFC_SW_DWN":  "ghi",
        "ALLSKY_SFC_SW_DNI":  "dni",
        "ALLSKY_SFC_SW_DIFF": "dhi",
        "T2M":                "temp_air",
        "WS10M":              "wind_speed",
    }
    pv_df = df.rename(columns=rename)
    required = ["ghi", "dni", "dhi", "temp_air", "wind_speed"]
    missing  = [c for c in required if c not in pv_df.columns]
    if missing:
        raise ValueError(f"Missing columns after rename: {missing}")

    pv_df = pv_df.dropna(subset=required)
    log.info(f"  {len(pv_df):,} rows after NaN drop")

    # ── pvlib Location + System ───────────────────────────────────────────────
    location = pvlib.location.Location(
        latitude=LATITUDE, longitude=LONGITUDE, tz=tz, altitude=ALTITUDE
    )
    module_params   = pvlib.pvsystem.retrieve_sam("CECMod")[MODULE_NAME]
    inverter_params = pvlib.pvsystem.retrieve_sam("CECInverter")[INVERTER_NAME]
    temp_params     = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]

    system = pvlib.pvsystem.PVSystem(
        surface_tilt=SURFACE_TILT,
        surface_azimuth=SURFACE_AZIMUTH,
        module_parameters=module_params,
        inverter_parameters=inverter_params,
        temperature_model_parameters=temp_params,
        modules_per_string=MODULES_PER_STRING,
        strings_per_inverter=STRINGS_PER_INV,
    )

    mc = pvlib.modelchain.ModelChain(
        system, location,
        aoi_model="physical",
        spectral_model="no_loss",
    )

    log.info("Running ModelChain simulation …")
    mc.run_model(pv_df)
    ac = mc.results.ac.fillna(0).clip(lower=0)
    ac.name = "simulated_ac_power_W"
    log.info(f"  Peak AC power: {ac.max():.0f} W")
    log.info(f"  Annual generation: {ac.sum()/1000:.1f} kWh")

    # ── Shading factor ────────────────────────────────────────────────────────
    log.info("Computing horizon shading factors …")
    shading = compute_shading_loss(pv_df.index, LATITUDE, LONGITUDE)
    shading.name = "shading_factor"

    # Apply shading to AC output
    ac_shaded = (ac * shading).clip(lower=0)
    ac_shaded.name = "ac_power_shaded_W"

    # ── Combine and save ──────────────────────────────────────────────────────
    out = df.join(ac, how="inner")
    out = out.join(shading, how="inner")
    out = out.join(ac_shaded, how="inner")

    # System capacity for reference
    system_kwp = (MODULES_PER_STRING * STRINGS_PER_INV *
                  module_params["STC"]) / 1000
    out["system_kwp"] = system_kwp

    out.to_csv(output_csv)
    log.info(f"Saved {len(out):,} rows → {output_csv}")
    log.info(f"  System size: {system_kwp:.2f} kWp")
    log.info(f"  Annual shaded generation: {ac_shaded.sum()/1000:.1f} kWh")

    return out


if __name__ == "__main__":
    result = simulate()
    print(f"\n✓ Simulation complete")
    print(f"  Rows: {len(result):,}")
    print(f"  Annual generation (unshaded): {result['simulated_ac_power_W'].sum()/1000:.1f} kWh")
    print(f"  Annual generation (shaded):   {result['ac_power_shaded_W'].sum()/1000:.1f} kWh")
    print(f"  Avg shading loss: {(1 - result['shading_factor'].mean())*100:.1f}%")
    print(f"  Saved to: {OUTPUT_CSV}")
