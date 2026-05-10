"""
Stage 3 — LSTM Training
Trains a PyTorch LSTM on weather → AC power sequences.
Adds time features (hour, month) to improve seasonal accuracy.
Saves model weights + both scalers for inference.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import logging
import argparse
from pathlib import Path

BASE_DIR       = Path(__file__).resolve().parent
PROCESSED_CSV  = BASE_DIR / "data" / "processed" / "weather_and_simulated_hourly_power.csv"
MODEL_DIR      = BASE_DIR / "models"
PLOT_DIR       = BASE_DIR / "plots"
LOG_DIR        = BASE_DIR / "logs"
for d in [MODEL_DIR, PLOT_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

MODEL_PATH          = MODEL_DIR / "best_lstm_model_hourly.pth"
FEATURE_SCALER_PATH = MODEL_DIR / "feature_scaler_hourly.joblib"
TARGET_SCALER_PATH  = MODEL_DIR / "target_scaler_hourly.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage3_training.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("stage3")

np.random.seed(42)
torch.manual_seed(42)

FEATURE_COLS = [
    "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF",
    "T2M", "WS10M",
    "hour_sin", "hour_cos",   # time-of-day encoding
    "month_sin", "month_cos", # seasonal encoding
]
TARGET_COL = "ac_power_shaded_W"   # use shaded output if available, else simulated_ac_power_W


class SolarLSTM(nn.Module):
    def __init__(self, input_size: int, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour-of-day and month as sine/cosine so the model sees cyclical time."""
    idx = df.index
    if hasattr(idx, "hour"):
        df = df.copy()
        df["hour_sin"]  = np.sin(2 * np.pi * idx.hour / 24)
        df["hour_cos"]  = np.cos(2 * np.pi * idx.hour / 24)
        df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    return df


def make_sequences(X: np.ndarray, y: np.ndarray,
                   seq_len: int):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def train(args):
    # ── Load data ─────────────────────────────────────────────────────────────
    log.info(f"Loading {args.input}")
    df = pd.read_csv(args.input, index_col="Timestamp", parse_dates=True)
    df.sort_index(inplace=True)

    # Use shaded power if available, else fall back to unshaded simulation
    target = TARGET_COL if TARGET_COL in df.columns else "simulated_ac_power_W"
    log.info(f"Target column: {target}")

    df = add_time_features(df)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    log.info(f"Features ({len(feat_cols)}): {feat_cols}")

    df = df[feat_cols + [target]].dropna()
    log.info(f"Dataset: {len(df):,} rows")

    # ── Chronological split ───────────────────────────────────────────────────
    n = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.10)
    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train: n_train + n_val]
    test_df  = df.iloc[n_train + n_val:]
    log.info(f"Split → train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")

    # ── Scale ─────────────────────────────────────────────────────────────────
    feat_scaler = MinMaxScaler()
    tgt_scaler  = MinMaxScaler()

    X_train = feat_scaler.fit_transform(train_df[feat_cols])
    y_train = tgt_scaler.fit_transform(train_df[[target]])
    X_val   = feat_scaler.transform(val_df[feat_cols])
    y_val   = tgt_scaler.transform(val_df[[target]])
    X_test  = feat_scaler.transform(test_df[feat_cols])
    y_test  = tgt_scaler.transform(test_df[[target]])

    joblib.dump(feat_scaler, FEATURE_SCALER_PATH)
    joblib.dump(tgt_scaler,  TARGET_SCALER_PATH)
    log.info("Scalers saved.")

    # ── Sequences ─────────────────────────────────────────────────────────────
    Xtr, ytr = make_sequences(X_train, y_train, args.seq_len)
    Xv,  yv  = make_sequences(X_val,   y_val,   args.seq_len)
    Xte, yte = make_sequences(X_test,  y_test,  args.seq_len)
    log.info(f"Sequence shapes → Xtr:{Xtr.shape} Xv:{Xv.shape} Xte:{Xte.shape}")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    def make_loader(X, y, shuffle):
        ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(Xtr, ytr, True)
    val_loader   = make_loader(Xv,  yv,  False)
    test_loader  = make_loader(Xte, yte, False)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model     = SolarLSTM(len(feat_cols), args.hidden, args.layers, args.dropout).to(device)
    criterion = nn.HuberLoss()    # more robust to outliers than MSE
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val   = float("inf")
    no_improve = 0
    train_hist, val_hist = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = 0
        for bX, by in train_loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
        tl /= len(train_loader)

        model.eval()
        vl = 0
        with torch.no_grad():
            for bX, by in val_loader:
                vl += criterion(model(bX.to(device)), by.to(device)).item()
        vl /= len(val_loader)
        scheduler.step(vl)

        train_hist.append(tl)
        val_hist.append(vl)

        if epoch % 5 == 0 or epoch == 1:
            log.info(f"Epoch {epoch:3d}/{args.epochs} | train={tl:.5f} val={vl:.5f}")

        if vl < best_val:
            best_val   = vl
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                log.info(f"Early stop at epoch {epoch}")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for bX, by in test_loader:
            preds.extend(model(bX.to(device)).cpu().numpy())
            actuals.extend(by.numpy())

    preds   = tgt_scaler.inverse_transform(np.array(preds))
    actuals = tgt_scaler.inverse_transform(np.array(actuals))

    r2   = r2_score(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae  = mean_absolute_error(actuals, preds)
    log.info(f"Test  R²={r2:.4f}  RMSE={rmse:.1f}W  MAE={mae:.1f}W")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_hist, label="Train")
    axes[0].plot(val_hist,   label="Val")
    axes[0].set_title("Loss history"); axes[0].legend(); axes[0].grid(True)

    n_plot = min(1000, len(actuals))
    idx    = np.linspace(0, len(actuals)-1, n_plot, dtype=int)
    axes[1].scatter(actuals[idx], preds[idx], s=3, alpha=0.4)
    mn, mx = actuals.min(), actuals.max()
    axes[1].plot([mn, mx], [mn, mx], "r--")
    axes[1].set_xlabel("Actual (W)"); axes[1].set_ylabel("Predicted (W)")
    axes[1].set_title(f"Test predictions  R²={r2:.3f}")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "training_results.png", dpi=120)
    plt.close()
    log.info(f"Plot saved → plots/training_results.png")

    return {"r2": r2, "rmse": rmse, "mae": mae,
            "feat_cols": feat_cols, "seq_len": args.seq_len,
            "hidden": args.hidden, "layers": args.layers}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    default=str(PROCESSED_CSV))
    parser.add_argument("--seq_len",  type=int,   default=24)
    parser.add_argument("--epochs",   type=int,   default=60)
    parser.add_argument("--batch",    type=int,   default=128)
    parser.add_argument("--lr",       type=float, default=0.001)
    parser.add_argument("--patience", type=int,   default=12)
    parser.add_argument("--hidden",   type=int,   default=128)
    parser.add_argument("--layers",   type=int,   default=2)
    parser.add_argument("--dropout",  type=float, default=0.2)
    args = parser.parse_args()

    print("=== Stage 3: LSTM Training ===")
    metrics = train(args)
    print(f"\n✓ Training complete")
    print(f"  R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.1f}W  MAE={metrics['mae']:.1f}W")
    print(f"  Model → {MODEL_PATH}")
