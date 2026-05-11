import logging
import sys
import json
=======
import logging
import sys
>>>>>>> f8bfef9a65ffb87656ba2f335d8759fc324a2f7e
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from pathlib import Path
<<<<<<< HEAD
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
=======
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Any, Optional

# --- Configuration ---
# Define paths relative to this script file
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# Paths to saved artifacts from training
MODEL_PATH = MODEL_DIR / "best_lstm_model_hourly.pth"
FEATURE_SCALER_PATH = MODEL_DIR / "feature_scaler_hourly.joblib"
TARGET_SCALER_PATH = MODEL_DIR / "target_scaler_hourly.joblib"
LOG_FILE_PATH = LOG_DIR / "app_hourly.log"

# --- Parameters (MUST MATCH TRAINING CONFIGURATION) ---
# These should ideally come from a shared config file used by both train and app
SEQUENCE_LENGTH = 24 # The sequence length used during training (e.g., args.seq_len in train_lstm.py)
# The exact feature columns used for training (order might matter depending on scaler)
FEATURE_COLS = ["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF", "T2M", "WS10M"] # MUST match args.feature_cols in train_lstm.py
INPUT_SIZE = len(FEATURE_COLS) # Automatically determined
HIDDEN_SIZE = 64 # MUST match args.hidden_size
NUM_LAYERS = 2   # MUST match args.num_layers
DROPOUT = 0.2    # MUST match args.dropout

# --- Create Directories ---
LOG_DIR.mkdir(exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w'),
        logging.StreamHandler(sys.stdout) # Also log to console
    ]
)
logger = logging.getLogger("PV_Prediction_API")

# --- LSTM Model Definition ---
# !! IDEALLY: Move this class to a separate 'model.py' and import it !!
class SolarLSTM(nn.Module):
    """ LSTM model for solar power prediction. """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2, output_size: int = 1):
        super(SolarLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        return out

# --- Global Variables for Loaded Objects ---
MODEL: Optional[SolarLSTM] = None
FEATURE_SCALER: Optional[MinMaxScaler] = None # Specify actual type if possible
TARGET_SCALER: Optional[MinMaxScaler] = None # Specify actual type if possible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "CPU")

# --- Loading Function ---
def load_artifacts():
    """Loads the model and scalers into global variables."""
    global MODEL, FEATURE_SCALER, TARGET_SCALER
    logger.info(f"Using device: {DEVICE}")

    # Load Feature Scaler
    try:
        if not FEATURE_SCALER_PATH.exists():
            raise FileNotFoundError(f"Feature scaler not found at {FEATURE_SCALER_PATH}")
        FEATURE_SCALER = joblib.load(FEATURE_SCALER_PATH)
        logger.info("Feature scaler loaded successfully.")
        # Simple check on the scaler type or attributes if possible
        if not hasattr(FEATURE_SCALER, 'transform'):
             logger.warning("Loaded feature scaler might be invalid (missing transform method).")

    except Exception as e:
        logger.critical(f"Failed to load feature scaler: {e}", exc_info=True)
        FEATURE_SCALER = None

    # Load Target Scaler
    try:
        if not TARGET_SCALER_PATH.exists():
            raise FileNotFoundError(f"Target scaler not found at {TARGET_SCALER_PATH}")
        TARGET_SCALER = joblib.load(TARGET_SCALER_PATH)
        logger.info("Target scaler loaded successfully.")
        if not hasattr(TARGET_SCALER, 'inverse_transform'):
             logger.warning("Loaded target scaler might be invalid (missing inverse_transform method).")

    except Exception as e:
        logger.critical(f"Failed to load target scaler: {e}", exc_info=True)
        TARGET_SCALER = None

    # Load Model
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        MODEL = SolarLSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )
        # Load state dict onto the correct device
        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        MODEL.to(DEVICE)
        MODEL.eval() # Set to evaluation mode
        logger.info("LSTM model loaded successfully.")

        # Optional: Perform a dummy inference pass to check
        try:
             dummy_input = torch.randn(1, SEQUENCE_LENGTH, INPUT_SIZE).to(DEVICE)
             with torch.no_grad():
                  _ = MODEL(dummy_input)
             logger.info("Dummy inference check passed.")
        except Exception as e_dummy:
             logger.warning(f"Dummy inference check failed: {e_dummy}")

    except Exception as e:
        logger.critical(f"Failed to load LSTM model: {e}", exc_info=True)
        MODEL = None

# --- Prediction Function ---
def predict_power(input_data: List[Dict[str, Any]]) -> float:
    """
    Preprocesses input data, runs prediction, and inverse transforms the result.

    Args:
        input_data: A list of dictionaries, where each dict represents an hour
                    and contains keys matching FEATURE_COLS. Must contain at
                    least SEQUENCE_LENGTH entries in chronological order.

    Returns:
        Predicted AC power in Watts.

    Raises:
        ValueError: If input data is invalid (wrong format, missing columns, not enough rows).
        RuntimeError: If model or scalers are not loaded.
    """
    if MODEL is None or FEATURE_SCALER is None or TARGET_SCALER is None:
        raise RuntimeError("Model or scalers are not loaded. Cannot predict.")

    # 1. Convert to DataFrame and Validate
    try:
        df = pd.DataFrame(input_data)
        if df.empty:
            raise ValueError("Input data is empty.")

        # Check for required feature columns
        missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}. Need: {FEATURE_COLS}")

        # Ensure data is numeric
        df = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
        if df.isnull().any().any():
             raise ValueError(f"Non-numeric data found in required columns after conversion: {df.isnull().sum()[df.isnull().sum()>0].index.tolist()}")

        # Check sequence length
        if len(df) < SEQUENCE_LENGTH:
            raise ValueError(f"Input data must contain at least {SEQUENCE_LENGTH} time steps (hours). Found {len(df)}.")

        # Select the last sequence
        sequence_df = df.iloc[-SEQUENCE_LENGTH:]

    except Exception as e:
        logger.error(f"Error processing input DataFrame: {e}")
        raise ValueError(f"Invalid input data format or content: {e}")

    # 2. Scale Features
    try:
        features_scaled = FEATURE_SCALER.transform(sequence_df.values)
    except Exception as e:
        logger.error(f"Error applying feature scaler: {e}", exc_info=True)
        # This might happen if the number of columns doesn't match what the scaler expects
        raise RuntimeError(f"Feature scaling failed. Check input columns match training. Error: {e}")


    # 3. Create Tensor
    input_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(DEVICE) # Add batch dimension

    # 4. Predict
    try:
        with torch.no_grad():
            prediction_scaled = MODEL(input_tensor) # Shape (1, 1)
    except Exception as e:
        logger.error(f"Model inference failed: {e}", exc_info=True)
        raise RuntimeError(f"Prediction failed during model execution: {e}")


    # 5. Inverse Transform Prediction
    try:
        # Target scaler was fitted on a single column, so direct inverse transform works
        prediction_actual = TARGET_SCALER.inverse_transform(prediction_scaled.cpu().numpy())
        # Result is [[value]], extract the float
        final_prediction = float(prediction_actual[0, 0])
        # Ensure prediction is physically plausible (e.g., >= 0)
        final_prediction = max(0.0, final_prediction)

    except Exception as e:
        logger.error(f"Error applying target inverse transform: {e}", exc_info=True)
        raise RuntimeError(f"Inverse transformation of prediction failed: {e}")

    return final_prediction


# --- Flask App Initialization ---
app = Flask(__name__)

# Load artifacts when the application starts
load_artifacts()

# --- API Routes ---
@app.route('/', methods=['GET'])
def root():
    logger.info("Root route accessed.")
    return jsonify({"api_status": "running",
                    "message": "Solar Power Prediction API (Hourly)",
                    "model_loaded": MODEL is not None,
                    "scalers_loaded": FEATURE_SCALER is not None and TARGET_SCALER is not None
                    })

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    if MODEL is not None and FEATURE_SCALER is not None and TARGET_SCALER is not None:
        status = "ok"
        message = "Model and scalers loaded successfully."
    else:
        status = "error"
        errors = []
        if MODEL is None: errors.append("Model failed to load")
        if FEATURE_SCALER is None: errors.append("Feature scaler failed to load")
        if TARGET_SCALER is None: errors.append("Target scaler failed to load")
        message = ", ".join(errors) + "."

    logger.debug(f"Health check accessed. Status: {status}")
    return jsonify({
        "status": status,
        "message": message,
        "model_path": str(MODEL_PATH),
        "feature_scaler_path": str(FEATURE_SCALER_PATH),
        "target_scaler_path": str(TARGET_SCALER_PATH),
        "device": str(DEVICE)
    }), 200 if status == "ok" else 503 # Return 503 if not ready

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint. Expects JSON data like:
    {
        "data": [
            {"Timestamp": "...", "ALLSKY_SFC_SW_DWN": 500, "ALLSKY_SFC_SW_DNI": 400, ...}, # Hour -N
            ... (at least SEQUENCE_LENGTH entries) ...
            {"Timestamp": "...", "ALLSKY_SFC_SW_DWN": 600, "ALLSKY_SFC_SW_DNI": 500, ...}  # Hour -1 (most recent)
        ]
    }
    """
    logger.info("Received request on /predict")

    # Check if model/scalers are ready
    if MODEL is None or FEATURE_SCALER is None or TARGET_SCALER is None:
        logger.error("Prediction attempt failed: Model/Scalers not loaded.")
        return jsonify({"error": "Service Unavailable: Model or scalers not loaded."}), 503

    # Get input data
    if not request.is_json:
        logger.warning("Request denied: Content-Type must be application/json.")
        return jsonify({"error": "Request must be JSON"}), 415

    req_data = request.get_json()
    if not isinstance(req_data, dict) or 'data' not in req_data:
        logger.warning("Request denied: JSON must be an object with a 'data' key.")
        return jsonify({"error": "Invalid JSON format: Missing 'data' key."}), 400

    input_sequence = req_data['data']
    if not isinstance(input_sequence, list):
        logger.warning("Request denied: 'data' field must be a list.")
        return jsonify({"error": "Invalid JSON format: 'data' must be a list of objects."}), 400

    # Perform prediction
    try:
        prediction = predict_power(input_sequence)
        logger.info(f"Prediction successful: {prediction:.2f} W")
        return jsonify({"predicted_power_W": prediction})

    except ValueError as e:
        logger.warning(f"Bad Request: Invalid input data - {e}")
        return jsonify({"error": f"Invalid Input Data: {e}"}), 400
    except RuntimeError as e:
        logger.error(f"Internal Server Error: Prediction runtime failed - {e}", exc_info=True)
        return jsonify({"error": f"Prediction Failed: {e}"}), 500
    except Exception as e:
        logger.error(f"Internal Server Error: Unexpected error during prediction - {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500

# --- Run the App ---
if __name__ == '__main__':
    host = '127.0.0.1' # Localhost
    port = 5001        # Choose a port
    logger.info(f"Starting Flask server on http://{host}:{port}")
    # Run in debug mode for development (auto-reloads on code change)
    # Set debug=False for production
    app.run(host=host, port=port, debug=True)
