"""
Stage 7 — Solar ROI Prediction API
Upgraded Flask API exposing:
  GET  /health          — service status
  POST /predict         — single next-hour power (W)
  POST /roi-report      — full 10-year ROI analysis
  GET  /tariffs         — list supported cities + tariffs
"""

import logging
import sys
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from pathlib import Path
from typing import Optional

BASE_DIR    = Path(__file__).resolve().parent
MODEL_DIR   = BASE_DIR / "models"
DATA_DIR    = BASE_DIR / "data"
LOG_DIR     = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

MODEL_PATH          = MODEL_DIR / "best_lstm_model_hourly.pth"
FEATURE_SCALER_PATH = MODEL_DIR / "feature_scaler_hourly.joblib"
TARGET_SCALER_PATH  = MODEL_DIR / "target_scaler_hourly.joblib"
FORECAST_CSV        = DATA_DIR  / "processed" / "monthly_forecast_10yr.csv"

SEQUENCE_LENGTH = 24
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
DROPOUT         = 0.2

FEATURE_COLS = [
    "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF",
    "T2M", "WS10M",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "api.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("api")


# ─── Model definition (must match training) ────────────────────────────────────
class SolarLSTM(nn.Module):
    def __init__(self, input_size, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ─── India tariff database ─────────────────────────────────────────────────────
STATE_TARIFFS = {
    "delhi": 8.50, "mumbai": 9.25, "bangalore": 7.10,
    "hyderabad": 6.30, "chennai": 5.80, "kolkata": 7.00,
    "pune": 9.00, "ahmedabad": 5.50, "jaipur": 6.80,
    "lucknow": 6.50, "chandigarh": 4.90, "bhopal": 7.20,
    "generic": 7.00,
}


def pm_surya_ghar_subsidy(kwp: float) -> float:
    if kwp <= 1:   return 30_000
    elif kwp <= 2: return 60_000
    elif kwp <= 3: return 78_000
    else:          return 78_000 + min(kwp - 3, 7) * 9_000


# ─── Global loaded objects ─────────────────────────────────────────────────────
MODEL         = None
FEAT_SCALER   = None
TGT_SCALER    = None
DEVICE        = torch.device("cpu")
INPUT_SIZE    = None  # will be set after loading scaler


def load_artifacts():
    global MODEL, FEAT_SCALER, TGT_SCALER, DEVICE, INPUT_SIZE

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {DEVICE}")

    try:
        FEAT_SCALER = joblib.load(FEATURE_SCALER_PATH)
        INPUT_SIZE  = FEAT_SCALER.n_features_in_
        log.info(f"Feature scaler loaded — {INPUT_SIZE} features")
    except Exception as e:
        log.critical(f"Feature scaler load failed: {e}")
        FEAT_SCALER = None

    try:
        TGT_SCALER = joblib.load(TARGET_SCALER_PATH)
        log.info("Target scaler loaded")
    except Exception as e:
        log.critical(f"Target scaler load failed: {e}")
        TGT_SCALER = None

    try:
        MODEL = SolarLSTM(INPUT_SIZE or 5, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        MODEL.to(DEVICE)
        MODEL.eval()
        # Dummy pass to warm up
        dummy = torch.zeros(1, SEQUENCE_LENGTH, INPUT_SIZE or 5).to(DEVICE)
        with torch.no_grad():
            MODEL(dummy)
        log.info("LSTM model loaded and warm-up passed")
    except Exception as e:
        log.critical(f"Model load failed: {e}")
        MODEL = None


# ─── Utility ──────────────────────────────────────────────────────────────────
def _compute_irr(cashflows, guess=0.01, max_iter=1000, tol=1e-6):
    r = guess
    for _ in range(max_iter):
        npv  = sum(cf / (1 + r) ** t for t, cf in enumerate(cashflows))
        dnpv = sum(-t * cf / (1 + r) ** (t + 1)
                   for t, cf in enumerate(cashflows) if t > 0)
        if abs(dnpv) < 1e-12:
            break
        r_new = r - npv / dnpv
        if abs(r_new - r) < tol:
            return r_new
        r = r_new
    return r if -0.05 < r < 0.5 else None


def _predict_one(data_list: list) -> float:
    """Predict next-hour power from a list of hourly dicts."""
    df = pd.DataFrame(data_list)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    # Add time features if timestamp present
    if "Timestamp" in df.columns or "timestamp" in df.columns:
        ts_col = "Timestamp" if "Timestamp" in df.columns else "timestamp"
        df.index = pd.to_datetime(df[ts_col])
        df["hour_sin"]  = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"]  = np.cos(2 * np.pi * df.index.hour / 24)
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
        feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    if len(df) < SEQUENCE_LENGTH:
        raise ValueError(f"Need {SEQUENCE_LENGTH} hours, got {len(df)}")

    seq_df = df.iloc[-SEQUENCE_LENGTH:][feat_cols[:INPUT_SIZE]]
    X      = FEAT_SCALER.transform(seq_df.values)
    tensor = torch.FloatTensor(X).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_scaled = MODEL(tensor).cpu().numpy()

    pred_w = float(TGT_SCALER.inverse_transform(pred_scaled)[0, 0])
    return max(0.0, pred_w)


def _run_roi(body: dict) -> dict:
    """
    Full ROI analysis from API request body.
    Required fields: system_kwp, install_cost_rs, monthly_usage_kwh, city
    Optional: net_metering_pct, annual_om_rs, tariff_escalation
    """
    system_kwp       = float(body.get("system_kwp", 3))
    install_cost     = float(body.get("install_cost_rs", 135_000))
    monthly_usage    = float(body.get("monthly_usage_kwh", 300))
    city             = str(body.get("city", "delhi")).lower().strip()
    net_metering_pct = float(body.get("net_metering_pct", 0.80))
    annual_om        = float(body.get("annual_om_rs", system_kwp * 750))
    tariff_escalation = float(body.get("tariff_escalation", 0.06))
    tariff           = STATE_TARIFFS.get(city, STATE_TARIFFS["generic"])

    subsidy  = pm_surya_ghar_subsidy(system_kwp)
    net_cost = max(0, install_cost - subsidy)

    # Load forecast
    if not FORECAST_CSV.exists():
        raise FileNotFoundError(
            "monthly_forecast_10yr.csv not found. Run stage4_forecast_10yr.py first."
        )
    forecast = pd.read_csv(FORECAST_CSV)

    monthly_rows     = []
    cumulative       = 0.0
    payback_month    = None
    cashflows        = [-net_cost]

    for _, row in forecast.iterrows():
        mo_idx = int(row["month_idx"])
        yr_num = int(row["year"])
        kwh    = float(row["kwh_degraded"])

        tariff_yr     = tariff * ((1 + tariff_escalation) ** (yr_num - 1))
        solar_covers  = min(kwh, monthly_usage)
        excess_kwh    = max(0, kwh - monthly_usage)
        grid_savings  = solar_covers * tariff_yr
        export_credit = excess_kwh * tariff_yr * net_metering_pct
        om_month      = annual_om / 12
        net_monthly   = grid_savings + export_credit - om_month
        cumulative   += net_monthly

        if payback_month is None and cumulative >= net_cost:
            payback_month = mo_idx

        cashflows.append(net_monthly)
        monthly_rows.append({
            "month_idx":       mo_idx,
            "year":            yr_num,
            "month":           int(row["month"]),
            "kwh_generated":   round(kwh, 2),
            "tariff":          round(tariff_yr, 4),
            "grid_savings_rs": round(grid_savings, 2),
            "export_credit_rs":round(export_credit, 2),
            "net_monthly_rs":  round(net_monthly, 2),
            "cumulative_rs":   round(cumulative, 2),
        })

    net_profit    = cumulative - net_cost
    irr_monthly   = _compute_irr(cashflows)
    irr_annual    = ((1 + irr_monthly) ** 12 - 1) * 100 if irr_monthly else None
    payback_years = round(payback_month / 12, 2) if payback_month else None

    total_kwh = sum(r["kwh_generated"] for r in monthly_rows)

    return {
        "input": {
            "system_kwp":       system_kwp,
            "install_cost_rs":  install_cost,
            "city":             city,
            "base_tariff":      tariff,
            "monthly_usage_kwh": monthly_usage,
        },
        "subsidy": {
            "pm_surya_ghar_rs": subsidy,
            "net_investment_rs": net_cost,
        },
        "result": {
            "payback_month":      payback_month,
            "payback_years":      payback_years,
            "payback_readable":   f"{int(payback_years)} years {round((payback_years % 1)*12)} months"
                                  if payback_years else "Not within 10 years",
            "net_profit_10yr_rs": round(net_profit, 0),
            "irr_annual_pct":     round(irr_annual, 2) if irr_annual else None,
            "total_10yr_kwh":     round(total_kwh, 0),
            "co2_saved_tonnes":   round(total_kwh * 0.82 / 1000, 2),
        },
        "monthly_cashflow": monthly_rows,
    }


# ─── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
load_artifacts()


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "Solar ROI Prediction API v2",
        "endpoints": {
            "GET  /health":     "Service health check",
            "POST /predict":    "Next-hour AC power (W)",
            "POST /roi-report": "Full 10-year ROI analysis",
            "GET  /tariffs":    "Supported cities and tariffs",
        },
        "model_ready": MODEL is not None,
    })


@app.route("/health", methods=["GET"])
def health():
    ok = MODEL is not None and FEAT_SCALER is not None and TGT_SCALER is not None
    return jsonify({
        "status": "ok" if ok else "degraded",
        "model":          MODEL is not None,
        "feat_scaler":    FEAT_SCALER is not None,
        "tgt_scaler":     TGT_SCALER is not None,
        "forecast_ready": FORECAST_CSV.exists(),
        "device":         str(DEVICE),
    }), 200 if ok else 503


@app.route("/tariffs", methods=["GET"])
def tariffs():
    return jsonify({
        "tariffs_rs_per_kwh": STATE_TARIFFS,
        "note": "FY 2024-25 residential DISCOM rates",
        "pm_surya_ghar_subsidy_slabs": {
            "upto_1kWp": 30_000,
            "1_to_2kWp": 60_000,
            "2_to_3kWp": 78_000,
            "3_to_10kWp": "78000 + 9000 per additional kWp",
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: { "data": [ { "ALLSKY_SFC_SW_DWN": 450, "T2M": 28, ... }, ... 24 items ] }
    Returns: { "predicted_power_W": 1847.3, "unit": "Watts" }
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    if MODEL is None or FEAT_SCALER is None or TGT_SCALER is None:
        return jsonify({"error": "Model not loaded"}), 503

    body = request.get_json()
    data = body.get("data")
    if not isinstance(data, list):
        return jsonify({"error": "'data' must be a list of hourly weather dicts"}), 400

    try:
        power_w = _predict_one(data)
        return jsonify({
            "predicted_power_W": round(power_w, 2),
            "predicted_power_kW": round(power_w / 1000, 4),
            "unit": "Watts",
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        log.error(f"Predict error: {e}", exc_info=True)
        return jsonify({"error": "Internal prediction error"}), 500


@app.route("/roi-report", methods=["POST"])
def roi_report():
    """
    POST /roi-report
    Body:
    {
      "system_kwp":         3.0,
      "install_cost_rs":    135000,
      "monthly_usage_kwh":  300,
      "city":               "delhi",
      "net_metering_pct":   0.80,     (optional)
      "tariff_escalation":  0.06      (optional)
    }
    Returns full ROI analysis JSON.
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    body = request.get_json()
    required = ["system_kwp", "install_cost_rs", "monthly_usage_kwh", "city"]
    missing  = [f for f in required if f not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        result = _run_roi(body)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e),
                        "hint": "Run stage4_forecast_10yr.py to generate the forecast first"}), 503
    except Exception as e:
        log.error(f"ROI error: {e}", exc_info=True)
        return jsonify({"error": "Internal ROI calculation error", "detail": str(e)}), 500


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5001
    log.info(f"Starting Solar ROI API v2 on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
