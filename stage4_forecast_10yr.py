import numpy as np
import pandas as pd
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
PLOT_DIR    = BASE_DIR / "plots"
LOG_DIR     = BASE_DIR / "logs"
for d in [PLOT_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

PROCESSED_CSV = DATA_DIR / "processed" / "weather_and_simulated_hourly_power.csv"
FORECAST_CSV  = DATA_DIR / "processed" / "monthly_forecast_10yr.csv"

DEGRADATION_RATE = 0.005

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage4_forecast.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("stage4")

def build_monthly_forecast(system_kwp=None):
    log.info(f"Loading {PROCESSED_CSV}")
    df = pd.read_csv(PROCESSED_CSV, index_col="Timestamp", parse_dates=True)
    df.sort_index(inplace=True)
    log.info(f"  {len(df):,} hourly rows, range: {df.index.min()} to {df.index.max()}")

    power_col = "ac_power_shaded_W" if "ac_power_shaded_W" in df.columns else "simulated_ac_power_W"
    log.info(f"Using column: {power_col}")

    hourly_w = df[power_col].clip(lower=0)

    monthly_kwh = hourly_w.resample("ME").sum() / 1000.0
    log.info(f"  Monthly kWh stats: min={monthly_kwh.min():.0f}, max={monthly_kwh.max():.0f}, "
             f"mean={monthly_kwh.mean():.0f} over {len(monthly_kwh)} months")

    if len(monthly_kwh) < 12:
        log.warning(f"Only {len(monthly_kwh)} months of data. "
                     "Less than 1 year, multiple years needed for reliable profiles.")

    days_per_month = {1:31,2:28,3:31,4:30,5:31,6:30,
                      7:31,8:31,9:30,10:31,11:30,12:31}

    monthly_profile = {}
    for mo in range(1, 13):
        mo_data = monthly_kwh[monthly_kwh.index.month == mo]
        if len(mo_data) > 0:
            monthly_profile[mo] = mo_data.mean()
        else:
            hourly_mo = hourly_w[hourly_w.index.month == mo]
            daily_avg = hourly_mo.groupby(hourly_mo.index.day).sum().mean() / 1000.0
            monthly_profile[mo] = daily_avg * days_per_month[mo]

    log.info(f"  Monthly profile (kWh): {[f'{m}: {v:.0f}' for m, v in monthly_profile.items()]}")

    if system_kwp is None and "system_kwp" in df.columns:
        system_kwp = df["system_kwp"].iloc[0]
    log.info(f"  System size: {system_kwp:.2f} kWp")

    forecast = []
    for yr in range(1, 11):
        degrade = (1 - DEGRADATION_RATE) ** (yr - 1)
        for mo in range(1, 13):
            kwh = monthly_profile[mo]
            forecast.append({
                "year":          yr,
                "month":         mo,
                "month_idx":     (yr - 1) * 12 + mo,
                "kwh":           round(kwh, 2),
                "kwh_degraded":  round(kwh * degrade, 2),
                "degrade_factor": round(degrade, 4),
            })
    forecast = pd.DataFrame(forecast)
    forecast["system_kwp"] = system_kwp or 3.0

    forecast.to_csv(FORECAST_CSV, index=False)
    log.info(f"Forecast saved -> {FORECAST_CSV}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].bar(forecast["month_idx"], forecast["kwh"],
                color="steelblue", alpha=0.6, label="No degradation")
    axes[0].bar(forecast["month_idx"], forecast["kwh_degraded"],
                color="orange", alpha=0.8, label="With degradation")
    axes[0].set_xlabel("Month (1=Jan yr1, 120=Dec yr10)")
    axes[0].set_ylabel("kWh")
    axes[0].set_title("10-Year Monthly Generation Forecast")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    for yr in range(1, 11):
        axes[0].axvline((yr-1)*12 + 0.5, color="gray", linewidth=0.5, linestyle="--")
        axes[0].text((yr-1)*12 + 6, axes[0].get_ylim()[1]*0.95,
                     f"Yr{yr}", ha="center", fontsize=8, color="gray")

    annual = forecast.groupby("year")[["kwh", "kwh_degraded"]].sum()
    x = annual.index
    axes[1].bar(x - 0.2, annual["kwh"],         0.4, label="No degradation", color="steelblue", alpha=0.7)
    axes[1].bar(x + 0.2, annual["kwh_degraded"], 0.4, label="With degradation", color="orange", alpha=0.8)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Annual kWh")
    axes[1].set_title("Annual Generation (10-Year Projection)")
    axes[1].set_xticks(x)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "forecast_10yr.png", dpi=120)
    plt.close()
    log.info("Plot saved -> plots/forecast_10yr.png")

    return forecast

if __name__ == "__main__":
    print("=== Stage 4: 10-Year Generation Forecast ===")
    fc = build_monthly_forecast()
    annual_totals = fc.groupby("year")["kwh_degraded"].sum()
    print(f"\nForecast complete - 10 years x 12 months")
    for yr, kwh in annual_totals.items():
        print(f"    Year {yr:2d}: {kwh:,.0f} kWh")
    print(f"\n  Saved to: {FORECAST_CSV}")
