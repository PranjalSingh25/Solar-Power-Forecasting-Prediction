import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from solar_common import SolarLSTM, FEATURE_COLS, add_time_features, SEQUENCE_LENGTH, HIDDEN_SIZE, NUM_LAYERS, DROPOUT

BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
MODEL_DIR   = BASE_DIR / "models"
PLOT_DIR    = BASE_DIR / "plots"
LOG_DIR     = BASE_DIR / "logs"
for d in [PLOT_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

PROCESSED_CSV       = DATA_DIR / "processed" / "weather_and_simulated_hourly_power.csv"
MODEL_PATH          = MODEL_DIR / "best_lstm_model_hourly.pth"
FEATURE_SCALER_PATH = MODEL_DIR / "feature_scaler_hourly.joblib"
TARGET_SCALER_PATH  = MODEL_DIR / "target_scaler_hourly.joblib"
FORECAST_CSV        = DATA_DIR / "processed" / "monthly_forecast_10yr.csv"

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
    df = add_time_features(df)

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning(f"Missing feature cols (will be skipped): {missing}")

    df = df.dropna(subset=feat_cols)
    log.info(f"  {len(df):,} rows after NaN drop")

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_scaler = joblib.load(FEATURE_SCALER_PATH)
    tgt_scaler  = joblib.load(TARGET_SCALER_PATH)

    n_feat_scaler = feat_scaler.n_features_in_
    feat_cols_use = feat_cols[:n_feat_scaler]

    log.info(f"Scaler expects {n_feat_scaler} features. Using: {feat_cols_use}")

    model = SolarLSTM(n_feat_scaler, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    log.info("Model loaded.")

    log.info("Predicting hourly generation ...")
    X_all  = feat_scaler.transform(df[feat_cols_use].values)
    preds  = np.full(len(df), np.nan)
    with torch.no_grad():
        for i in range(SEQUENCE_LENGTH, len(X_all)):
            seq   = torch.FloatTensor(X_all[i - SEQUENCE_LENGTH: i]).unsqueeze(0).to(device)
            p     = model(seq).cpu().numpy()
            preds[i] = tgt_scaler.inverse_transform(p)[0, 0]
    preds = np.clip(preds, 0, None)
    pred_series = pd.Series(preds, index=df.index, name="predicted_W")
    log.info(f"  Predicted non-NaN values: {(~np.isnan(preds)).sum():,}")

    days_per_month = {1:31,2:28,3:31,4:30,5:31,6:30,
                      7:31,8:31,9:30,10:31,11:30,12:31}
    monthly_kwh = pd.Series(
        {m: (pred_series.dropna()[pred_series.dropna().index.month == m].mean()
             * 24 * days_per_month.get(m, 30) / 1000)
         for m in range(1, 13)},
        name="kwh"
    )
    log.info(f"  Monthly kWh range: {monthly_kwh.min():.0f}-{monthly_kwh.max():.0f}")

    rows = []
    for yr in range(1, 11):
        degrade = (1 - DEGRADATION_RATE) ** (yr - 1)
        for mo in range(1, 13):
            kwh = monthly_kwh[mo]
            rows.append({
                "year":        yr,
                "month":       mo,
                "month_idx":   (yr - 1) * 12 + mo,
                "kwh":         round(kwh, 2),
                "kwh_degraded": round(kwh * degrade, 2),
                "degrade_factor": round(degrade, 4),
            })
    forecast = pd.DataFrame(rows)

    if system_kwp is None and "system_kwp" in df.columns:
        system_kwp = df["system_kwp"].iloc[0]
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
    print(f"\n  Annual generation (with degradation):")
    for yr, kwh in annual_totals.items():
        print(f"    Year {yr:2d}: {kwh:,.0f} kWh")
    print(f"\n  Saved to: {FORECAST_CSV}")
