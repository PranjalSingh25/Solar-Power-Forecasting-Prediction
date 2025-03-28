import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import logging
import joblib # <-- For saving scalers
import argparse
from pathlib import Path
from typing import List, Tuple, Optional # Added Optional

# --- Configuration & Setup ---
# Use Path for better cross-platform compatibility
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PLOT_DIR = BASE_DIR / "plots"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [PLOT_DIR, MODEL_DIR, LOGS_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(exist_ok=True)

# Input file from the simulation step
INPUT_DATA_FILE = PROCESSED_DATA_DIR / "weather_and_simulated_hourly_power.csv"

# Output files
MODEL_SAVE_PATH = MODEL_DIR / "best_lstm_model_hourly.pth"
FEATURE_SCALER_PATH = MODEL_DIR / "feature_scaler_hourly.joblib"
TARGET_SCALER_PATH = MODEL_DIR / "target_scaler_hourly.joblib"
TRAINING_PLOT_PATH = PLOT_DIR / "training_history_hourly.png"
TEST_PRED_PLOT_PATH = PLOT_DIR / "test_predictions_hourly.png"

LOG_FILE = LOGS_DIR / "lstm_training_hourly.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lstm_training_hourly")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- LSTM Model Definition ---
class SolarLSTM(nn.Module):
    """
    LSTM model for solar power prediction based on weather features.
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2, output_size: int = 1):
        super(SolarLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # Input: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0
        )

        # Using a simpler final layer, adjust if needed
        self.fc = nn.Linear(hidden_size, output_size)
        # Consider adding BatchNorm/Dropout between LSTM and FC if overfitting occurs

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use the output of the last time step for sequence-to-value prediction
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        return out

# --- Data Handling Functions ---

def load_processed_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Loads the processed CSV file containing weather and simulated power."""
    logger.info(f"Loading processed data from: {file_path}")
    try:
        if not file_path.exists():
            logger.error(f"Input data file not found: {file_path}")
            return None
        # Assuming the index column is named 'Timestamp' from the previous script
        df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
        df.sort_index(inplace=True) # Ensure chronological order

        # Basic check for expected columns
        expected_cols = ['simulated_ac_power_W', 'T2M', 'WS10M'] # Add others used as features
        if not all(col in df.columns for col in expected_cols):
             logger.warning(f"Loaded data might be missing expected columns. Found: {df.columns.tolist()}")

        # Handle potential remaining NaNs (e.g., from simulation issues)
        initial_len = len(df)
        df.dropna(inplace=True) # Simple drop for now, consider interpolation if appropriate
        if len(df) < initial_len:
             logger.warning(f"Dropped {initial_len - len(df)} rows containing NaNs after loading.")

        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}", exc_info=True)
        return None

def create_lstm_sequences(features_scaled: np.ndarray, target_scaled: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Creates sequences for LSTM input and target."""
    X, y = [], []
    if len(features_scaled) <= sequence_length:
        raise ValueError("Not enough data to create sequences with the specified length.")

    for i in range(len(features_scaled) - sequence_length):
        X.append(features_scaled[i:i + sequence_length])
        y.append(target_scaled[i + sequence_length]) # Predict the step right after the sequence

    return np.array(X), np.array(y)

# --- Training Function ---

def train_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int = 24, # e.g., use 24 hours of past data
    split_ratio_train: float = 0.7,
    split_ratio_val: float = 0.1, # Train/Val/Test split adds up to 1.0
    epochs: int = 50,
    batch_size: int = 64,
    patience: int = 10,
    learning_rate: float = 0.001,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2
    ) -> bool:
    """
    Trains the LSTM model, saves the best model and scalers, performs test evaluation.
    """
    logger.info("Starting model training process...")
    try:
        # --- 1. Feature and Target Selection ---
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in DataFrame.")
            return False
        missing_features = [f for f in feature_cols if f not in df.columns]
        if missing_features:
            logger.error(f"Feature columns not found in DataFrame: {missing_features}")
            return False

        features = df[feature_cols]
        target = df[[target_col]] # Keep as DataFrame for consistency

        # --- 2. Chronological Data Splitting ---
        n_samples = len(features)
        n_train = int(n_samples * split_ratio_train)
        n_val = int(n_samples * split_ratio_val)
        n_test = n_samples - n_train - n_val

        if n_train <= sequence_length or n_val <= sequence_length or n_test <= sequence_length:
             logger.error(f"Dataset too small for sequence length {sequence_length} and splits. "
                          f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
             return False

        train_features, train_target = features[:n_train], target[:n_train]
        val_features, val_target = features[n_train:n_train + n_val], target[n_train:n_train + n_val]
        test_features, test_target = features[n_train + n_val:], target[n_train + n_val:]

        logger.info(f"Data split - Train: {len(train_features)}, Validation: {len(val_features)}, Test: {len(test_features)}")

        # --- 3. Scaling ---
        logger.info("Scaling data...")
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        # Fit scalers ONLY on training data
        train_features_scaled = feature_scaler.fit_transform(train_features)
        train_target_scaled = target_scaler.fit_transform(train_target)

        # Transform validation and test data using the fitted scalers
        val_features_scaled = feature_scaler.transform(val_features)
        val_target_scaled = target_scaler.transform(val_target)
        test_features_scaled = feature_scaler.transform(test_features)
        test_target_scaled = target_scaler.transform(test_target)

        # Save the scalers
        logger.info(f"Saving feature scaler to {FEATURE_SCALER_PATH}")
        joblib.dump(feature_scaler, FEATURE_SCALER_PATH)
        logger.info(f"Saving target scaler to {TARGET_SCALER_PATH}")
        joblib.dump(target_scaler, TARGET_SCALER_PATH)

        # --- 4. Create Sequences ---
        logger.info(f"Creating sequences with length {sequence_length}...")
        X_train, y_train = create_lstm_sequences(train_features_scaled, train_target_scaled, sequence_length)
        X_val, y_val = create_lstm_sequences(val_features_scaled, val_target_scaled, sequence_length)
        X_test, y_test = create_lstm_sequences(test_features_scaled, test_target_scaled, sequence_length)

        logger.info(f"Sequence shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"Sequence shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
        logger.info(f"Sequence shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")

        # --- 5. DataLoaders ---
        logger.info("Creating DataLoaders...")
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # shuffle=True is important for training
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0) # No shuffle for test

        # --- 6. Model, Loss, Optimizer ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        input_size = X_train.shape[2] # Number of features
        model = SolarLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        logger.info(f"Model initialized:\n{model}")

        # --- 7. Training Loop ---
        logger.info("Starting training loop...")
        best_val_loss = float('inf')
        epochs_no_improve = 0
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            model.train() # Set model to training mode
            total_train_loss = 0.0
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval() # Set model to evaluation mode
            total_val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            # Early Stopping and Model Saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                logger.info(f"Validation loss improved. Saving model to {MODEL_SAVE_PATH}")
            else:
                epochs_no_improve += 1
                logger.info(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

        logger.info("Training loop finished.")

        # --- 8. Plot Training History ---
        logger.info(f"Saving training history plot to {TRAINING_PLOT_PATH}")
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Model Training History (Loss vs. Epoch)')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(TRAINING_PLOT_PATH)
        plt.close()

        # --- 9. Final Test Set Evaluation ---
        logger.info("Performing final evaluation on the test set...")
        evaluate_model(
            model_path=MODEL_SAVE_PATH,
            feature_scaler_path=FEATURE_SCALER_PATH,
            target_scaler_path=TARGET_SCALER_PATH,
            test_loader=test_loader,
            device=device,
            input_size=input_size, # Pass model parameters needed for instantiation
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        return True

    except ValueError as ve: # Catch specific errors like insufficient data
         logger.error(f"ValueError during training setup: {ve}", exc_info=True)
         return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)
        return False

# --- Evaluation Function ---
def evaluate_model(
    model_path: Path,
    feature_scaler_path: Path,
    target_scaler_path: Path,
    test_loader: DataLoader,
    device: torch.device,
    input_size: int, # Need these to reconstruct the model
    hidden_size: int,
    num_layers: int,
    dropout: float
    ):
    """Loads the best model and evaluates it on the test set."""
    try:
        # Load scalers
        if not feature_scaler_path.exists() or not target_scaler_path.exists():
             logger.error("Scaler files not found for evaluation.")
             return
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        logger.info("Scalers loaded successfully.")

        # Load model
        if not model_path.exists():
             logger.error(f"Model file not found: {model_path}")
             return
        model = SolarLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Set to evaluation mode
        logger.info("Best model loaded successfully.")

        predictions_scaled = []
        actuals_scaled = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                predictions_scaled.extend(outputs.cpu().numpy())
                actuals_scaled.extend(batch_y.cpu().numpy())

        # Inverse transform
        predictions_actual = target_scaler.inverse_transform(np.array(predictions_scaled))
        actuals_actual = target_scaler.inverse_transform(np.array(actuals_scaled))

        # Calculate metrics
        mse = mean_squared_error(actuals_actual, predictions_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals_actual, predictions_actual)
        r2 = r2_score(actuals_actual, predictions_actual)

        logger.info("--- Test Set Evaluation Metrics ---")
        logger.info(f"  MSE: {mse:.4f}")
        logger.info(f" RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"   R²: {r2:.4f}")
        logger.info("-----------------------------------")

        # Plot predictions vs actuals (optional, sample subset for clarity)
        sample_size = min(len(actuals_actual), 1000) # Plot max 1000 points
        indices = np.linspace(0, len(actuals_actual) - 1, sample_size, dtype=int)

        plt.figure(figsize=(15, 7))
        plt.plot(actuals_actual[indices], label='Actual Power', marker='.', linestyle='None', alpha=0.7)
        plt.plot(predictions_actual[indices], label='Predicted Power', marker='x', linestyle='None', alpha=0.7)
        plt.title(f'Test Set: Actual vs. Predicted Power (Sample of {sample_size})')
        plt.xlabel('Sample Index (from Test Set)')
        plt.ylabel('Simulated AC Power (W)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(TEST_PRED_PLOT_PATH)
        plt.close()
        logger.info(f"Test prediction plot saved to {TEST_PRED_PLOT_PATH}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="LSTM Solar Power Prediction Training")

    # --- Arguments ---
    parser.add_argument(
        "--input_file", type=str, default=str(INPUT_DATA_FILE),
        help="Path to the processed input CSV file (weather + simulated power)"
    )
    parser.add_argument(
        "--feature_cols", nargs='+',
        default=["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF", "T2M", "WS10M"], # Example features
        help="List of column names to use as input features."
    )
    parser.add_argument(
        "--target_col", type=str, default="simulated_ac_power_W",
        help="Name of the target column to predict."
    )
    parser.add_argument(
        "--seq_len", type=int, default=24, # Default to 24 hours
        help="Sequence length (number of past time steps) for LSTM input."
    )
    parser.add_argument("--epochs", type=int, default=50, help="Max number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--hidden_size", type=int, default=64, help="Number of hidden units in LSTM.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--split_train", type=float, default=0.7, help="Fraction of data for training.")
    parser.add_argument("--split_val", type=float, default=0.1, help="Fraction of data for validation.")
    # Test split is automatically calculated (1 - train - val)

    args = parser.parse_args()

    # Validate splits
    if not (0 < args.split_train < 1 and 0 < args.split_val < 1 and (args.split_train + args.split_val) < 1):
         logger.error("Invalid split ratios. Train and Val splits must be between 0 and 1, and their sum must be less than 1.")
         return

    logger.info("--- Starting LSTM Training Pipeline ---")
    logger.info(f"Arguments: {args}")

    # Load data
    df = load_processed_data(Path(args.input_file))

    if df is not None and not df.empty:
        # Train model
        success = train_model(
            df=df,
            feature_cols=args.feature_cols,
            target_col=args.target_col,
            sequence_length=args.seq_len,
            split_ratio_train=args.split_train,
            split_ratio_val=args.split_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            learning_rate=args.lr,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

        if success:
            logger.info("--- LSTM Training Pipeline Completed Successfully ---")
        else:
            logger.error("--- LSTM Training Pipeline Failed ---")
    else:
        logger.error("Failed to load data. Aborting training.")


if __name__ == "__main__":
    main()