import logging
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from pathlib import Path
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Loading Function ---
def load_artifacts():
    """Loads the model and scalers into global variables."""
    global MODEL, FEATURE_SCALER, TARGET_SCALER, DEVICE
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