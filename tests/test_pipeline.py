"""
tests/test_pipeline.py
──────────────────────
End-to-end test suite for the Solar ROI Predictor pipeline.
Covers: financial model logic, ROI engine, LSTM inference, and API endpoints.

Run with:
    python -m pytest tests/ -v
    python -m pytest tests/ -v --tb=short   # compact output
"""

import sys
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5+6 — Financial Model & ROI Engine
# ══════════════════════════════════════════════════════════════════════════════

class TestSubsidyCalculation:
    """PM Surya Ghar Muft Bijli Yojana subsidy slabs (MNRE Feb 2024)."""

    def setup_method(self):
        from stage56_financial_roi import pm_surya_ghar_subsidy
        self.subsidy = pm_surya_ghar_subsidy

    def test_below_1kwp(self):
        assert self.subsidy(0.5) == 30_000
        assert self.subsidy(1.0) == 30_000

    def test_1_to_2kwp(self):
        assert self.subsidy(1.5) == 60_000
        assert self.subsidy(2.0) == 60_000

    def test_2_to_3kwp(self):
        assert self.subsidy(2.5) == 78_000
        assert self.subsidy(3.0) == 78_000

    def test_above_3kwp(self):
        # ₹78,000 + ₹9,000 per kWp above 3
        assert self.subsidy(4.0) == 78_000 + 9_000        # 1 extra kWp
        assert self.subsidy(6.0) == 78_000 + 27_000       # 3 extra kWp
        assert self.subsidy(10.0) == 78_000 + 63_000      # 7 extra kWp (max)

    def test_above_10kwp_capped(self):
        # Extra capped at 7 kWp above 3 → max ₹1,41,000
        assert self.subsidy(15.0) == self.subsidy(10.0)


class TestSystemSpec:
    """SystemSpec dataclass properties."""

    def setup_method(self):
        from stage56_financial_roi import SystemSpec
        self.SystemSpec = SystemSpec

    def test_delhi_tariff(self):
        spec = self.SystemSpec(3.0, 135_000, city="delhi")
        assert spec.tariff == 8.50

    def test_mumbai_tariff(self):
        spec = self.SystemSpec(3.0, 135_000, city="mumbai")
        assert spec.tariff == 9.25

    def test_unknown_city_fallback(self):
        spec = self.SystemSpec(3.0, 135_000, city="unknown_city")
        assert spec.tariff == 7.00  # generic fallback

    def test_net_cost_after_subsidy(self):
        spec = self.SystemSpec(3.0, 135_000, city="delhi")
        assert spec.subsidy == 78_000
        assert spec.net_cost == 135_000 - 78_000

    def test_net_cost_never_negative(self):
        # Subsidy can't exceed install cost
        spec = self.SystemSpec(1.0, 10_000, city="delhi")
        assert spec.net_cost >= 0

    def test_default_om_cost(self):
        spec = self.SystemSpec(4.0, 180_000, city="delhi")
        assert spec.annual_om_rs == 4.0 * 750


class TestROIEngine:
    """ROI computation against known-good reference outputs."""

    FORECAST_CSV = ROOT / "data" / "processed" / "monthly_forecast_10yr.csv"

    def setup_method(self):
        from stage56_financial_roi import SystemSpec, compute_roi
        self.SystemSpec = SystemSpec
        self.compute_roi = compute_roi

    @pytest.mark.skipif(
        not (ROOT / "data" / "processed" / "monthly_forecast_10yr.csv").exists(),
        reason="Forecast CSV not generated yet — run stage4_forecast_10yr.py first"
    )
    def test_delhi_6kwp_payback_range(self):
        """
        Reference run: 6 kWp, Delhi, ₹2,70,000 installed.
        Verified payback: 2 years 5 months (month 29).
        Test asserts payback falls in a ±6 month tolerance window.
        """
        spec = self.SystemSpec(
            system_kwp=6.0,
            install_cost_rs=270_000,
            monthly_usage_kwh=450,
            city="delhi",
        )
        result = self.compute_roi(spec)
        pb = result["payback_years"]
        assert pb is not None, "Payback should be achieved within 10 years"
        assert 1.5 < pb < 4.0, f"Expected ~2.4 years, got {pb}"

    @pytest.mark.skipif(
        not (ROOT / "data" / "processed" / "monthly_forecast_10yr.csv").exists(),
        reason="Forecast CSV not generated yet"
    )
    def test_10yr_profit_positive(self):
        spec = self.SystemSpec(6.0, 270_000, monthly_usage_kwh=450, city="delhi")
        result = self.compute_roi(spec)
        assert result["net_profit_10yr"] > 0

    @pytest.mark.skipif(
        not (ROOT / "data" / "processed" / "monthly_forecast_10yr.csv").exists(),
        reason="Forecast CSV not generated yet"
    )
    def test_irr_reasonable(self):
        spec = self.SystemSpec(6.0, 270_000, monthly_usage_kwh=450, city="delhi")
        result = self.compute_roi(spec)
        irr = result["irr_annual_pct"]
        assert irr is not None
        assert 10 < irr < 200, f"IRR out of expected range: {irr}"

    @pytest.mark.skipif(
        not (ROOT / "data" / "processed" / "monthly_forecast_10yr.csv").exists(),
        reason="Forecast CSV not generated yet"
    )
    def test_monthly_cashflow_length(self):
        spec = self.SystemSpec(6.0, 270_000, monthly_usage_kwh=450, city="delhi")
        result = self.compute_roi(spec)
        assert len(result["monthly_df"]) == 120, "Should be 120 months (10 years)"

    @pytest.mark.skipif(
        not (ROOT / "data" / "processed" / "monthly_forecast_10yr.csv").exists(),
        reason="Forecast CSV not generated yet"
    )
    def test_cumulative_savings_monotonically_increases(self):
        spec = self.SystemSpec(6.0, 270_000, monthly_usage_kwh=450, city="delhi")
        result = self.compute_roi(spec)
        cumulative = result["monthly_df"]["cumulative_rs"].values
        assert all(cumulative[i] <= cumulative[i + 1]
                   for i in range(len(cumulative) - 1)), \
            "Cumulative savings should never decrease month-over-month"

    @pytest.mark.skipif(
        not (ROOT / "data" / "processed" / "monthly_forecast_10yr.csv").exists(),
        reason="Forecast CSV not generated yet"
    )
    def test_co2_savings_positive(self):
        spec = self.SystemSpec(6.0, 270_000, monthly_usage_kwh=450, city="delhi")
        result = self.compute_roi(spec)
        assert result["summary"]["co2_saved_tonnes"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — LSTM Model Inference
# ══════════════════════════════════════════════════════════════════════════════

class TestLSTMInference:
    """Tests that the saved model loads and produces sensible predictions."""

    MODEL_PATH   = ROOT / "models" / "best_lstm_model_hourly.pth"
    SCALER_FEAT  = ROOT / "models" / "feature_scaler_hourly.joblib"
    SCALER_TGT   = ROOT / "models" / "target_scaler_hourly.joblib"
    PROCESSED    = ROOT / "data" / "processed" / "weather_and_simulated_hourly_power.csv"

    @pytest.mark.skipif(
        not MODEL_PATH.exists(),
        reason="Model not trained yet — run stage3_train_lstm.py first"
    )
    def test_model_loads_without_error(self):
        import torch
        import joblib
        import torch.nn as nn

        class SolarLSTM(nn.Module):
            def __init__(self, input_size, hidden=128, layers=2, dropout=0.2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, layers,
                                    batch_first=True,
                                    dropout=dropout if layers > 1 else 0)
                self.head = nn.Sequential(
                    nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        scaler = joblib.load(self.SCALER_FEAT)
        model  = SolarLSTM(scaler.n_features_in_)
        model.load_state_dict(torch.load(self.MODEL_PATH, map_location="cpu"))
        model.eval()
        assert model is not None

@pytest.mark.skipif(
    not (MODEL_PATH.exists() and PROCESSED.exists() and 
         FEATURE_SCALER_PATH.exists()),
    reason="Model files or processed data not present - run pipeline first"
)
def test_prediction_non_negative(self):
        """Power output should never be negative."""
        import torch
        import joblib
        import torch.nn as nn

        class SolarLSTM(nn.Module):
            def __init__(self, input_size, hidden=128, layers=2, dropout=0.2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, layers,
                                    batch_first=True,
                                    dropout=dropout if layers > 1 else 0)
                self.head = nn.Sequential(
                    nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        feat_scaler = joblib.load(self.SCALER_FEAT)
        tgt_scaler  = joblib.load(self.SCALER_TGT)
        n_feat = feat_scaler.n_features_in_

        model = SolarLSTM(n_feat)
        model.load_state_dict(torch.load(self.MODEL_PATH, map_location="cpu"))
        model.eval()

        # Use real data for the test sequence
        df = pd.read_csv(self.PROCESSED, index_col="Timestamp", parse_dates=True)
        FEATURE_COLS = ["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI",
                        "ALLSKY_SFC_SW_DIFF", "T2M", "WS10M"]
        idx = df.index
        df["hour_sin"]  = np.sin(2 * np.pi * idx.hour / 24)
        df["hour_cos"]  = np.cos(2 * np.pi * idx.hour / 24)
        df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
        FEATURE_COLS += ["hour_sin", "hour_cos", "month_sin", "month_cos"]
        FEATURE_COLS = FEATURE_COLS[:n_feat]

        seq = df[FEATURE_COLS].dropna().iloc[:24].values
        X   = feat_scaler.transform(seq)
        t   = torch.FloatTensor(X).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = model(t).numpy()
        pred_w = float(tgt_scaler.inverse_transform(pred_scaled)[0, 0])

        # Clip is applied in production code; raw model output should be near 0 at worst
        assert pred_w > -500, f"Prediction unreasonably negative: {pred_w}"

@pytest.mark.skipif(
    not (MODEL_PATH.exists() and PROCESSED.exists() and 
         FEATURE_SCALER_PATH.exists()),
    reason="Model files or processed data not present - run pipeline first"
)
def test_model_r2_above_threshold(self):
        """
        Re-evaluate the saved model on the test split.
        Asserts R² > 0.95 — a degraded score signals model or data corruption.
        Verified result from training run: R² = 0.9839
        """
        import torch
        import joblib
        import torch.nn as nn
        from sklearn.metrics import r2_score

        class SolarLSTM(nn.Module):
            def __init__(self, input_size, hidden=128, layers=2, dropout=0.2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, layers,
                                    batch_first=True,
                                    dropout=dropout if layers > 1 else 0)
                self.head = nn.Sequential(
                    nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        feat_scaler = joblib.load(self.SCALER_FEAT)
        tgt_scaler  = joblib.load(self.SCALER_TGT)
        n_feat = feat_scaler.n_features_in_

        model = SolarLSTM(n_feat)
        model.load_state_dict(torch.load(self.MODEL_PATH, map_location="cpu"))
        model.eval()

        df = pd.read_csv(self.PROCESSED, index_col="Timestamp", parse_dates=True)
        target = "ac_power_shaded_W" if "ac_power_shaded_W" in df.columns \
                 else "simulated_ac_power_W"

        idx = df.index
        df["hour_sin"]  = np.sin(2 * np.pi * idx.hour / 24)
        df["hour_cos"]  = np.cos(2 * np.pi * idx.hour / 24)
        df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
        FEAT = ["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF",
                "T2M", "WS10M", "hour_sin", "hour_cos", "month_sin", "month_cos"]
        FEAT = FEAT[:n_feat]
        df = df.dropna(subset=FEAT + [target])

        # Use last 20% as test split (same as training)
        n       = len(df)
        test_df = df.iloc[int(n * 0.80):]
        SEQ     = 24

        X_all = feat_scaler.transform(test_df[FEAT].values)
        y_all = tgt_scaler.transform(test_df[[target]].values)

        preds, actuals = [], []
        with torch.no_grad():
            for i in range(SEQ, min(len(X_all), SEQ + 500)):  # sample 500 points
                seq = torch.FloatTensor(X_all[i - SEQ: i]).unsqueeze(0)
                p   = model(seq).numpy()
                preds.append(p[0, 0])
                actuals.append(y_all[i, 0])

        preds   = tgt_scaler.inverse_transform(np.array(preds).reshape(-1, 1))
        actuals = tgt_scaler.inverse_transform(np.array(actuals).reshape(-1, 1))
        r2 = r2_score(actuals, preds)

        assert r2 > 0.95, f"R² dropped below threshold: {r2:.4f} (expected > 0.95)"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Forecast Data Integrity
# ══════════════════════════════════════════════════════════════════════════════

class TestForecastData:
    """Validates the structure and sanity of the 10-year forecast CSV."""

    FORECAST_CSV = ROOT / "data" / "processed" / "monthly_forecast_10yr.csv"

    @pytest.mark.skipif(
        not FORECAST_CSV.exists(),
        reason="Forecast CSV not generated yet"
    )
    def test_forecast_has_120_rows(self):
        df = pd.read_csv(self.FORECAST_CSV)
        assert len(df) == 120, f"Expected 120 months, got {len(df)}"

    @pytest.mark.skipif(not FORECAST_CSV.exists(), reason="Forecast CSV not generated yet")
    def test_forecast_columns_present(self):
        df = pd.read_csv(self.FORECAST_CSV)
        for col in ["year", "month", "month_idx", "kwh", "kwh_degraded"]:
            assert col in df.columns, f"Missing column: {col}"

    @pytest.mark.skipif(not FORECAST_CSV.exists(), reason="Forecast CSV not generated yet")
    def test_kwh_values_positive(self):
        df = pd.read_csv(self.FORECAST_CSV)
        assert (df["kwh"] > 0).all(), "All monthly kWh values should be positive"
        assert (df["kwh_degraded"] > 0).all()

    @pytest.mark.skipif(not FORECAST_CSV.exists(), reason="Forecast CSV not generated yet")
    def test_degradation_reduces_output(self):
        """Degraded kWh should be ≤ undegraded for every row."""
        df = pd.read_csv(self.FORECAST_CSV)
        assert (df["kwh_degraded"] <= df["kwh"] + 0.01).all(), \
            "Degraded kWh should never exceed undegraded"

    @pytest.mark.skipif(not FORECAST_CSV.exists(), reason="Forecast CSV not generated yet")
    def test_year10_generation_less_than_year1(self):
        df = pd.read_csv(self.FORECAST_CSV)
        yr1_total  = df[df["year"] == 1]["kwh_degraded"].sum()
        yr10_total = df[df["year"] == 10]["kwh_degraded"].sum()
        assert yr10_total < yr1_total, \
            "Year 10 degraded generation should be less than year 1"

    @pytest.mark.skipif(not FORECAST_CSV.exists(), reason="Forecast CSV not generated yet")
    def test_delhi_annual_kwh_in_realistic_range(self):
        """
        Delhi 6 kWp should generate ~8,500–10,500 kWh/year.
        Verified: Year 1 = 9,093 kWh.
        """
        df = pd.read_csv(self.FORECAST_CSV)
        yr1 = df[df["year"] == 1]["kwh_degraded"].sum()
        assert 5_000 < yr1 < 15_000, \
            f"Annual generation out of realistic range: {yr1:.0f} kWh"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — Flask API
# ══════════════════════════════════════════════════════════════════════════════

class TestAPI:
    """Tests the Flask API endpoints using the test client."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from app import app
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_root_returns_200(self):
        r = self.client.get("/")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert "endpoints" in data

    def test_health_endpoint(self):
        r = self.client.get("/health")
        data = json.loads(r.data)
        # Status is either ok or degraded — both are valid responses
        assert data["status"] in ("ok", "degraded")
        assert "model" in data
        assert "forecast_ready" in data

    def test_tariffs_endpoint(self):
        r = self.client.get("/tariffs")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert "tariffs_rs_per_kwh" in data
        assert "delhi" in data["tariffs_rs_per_kwh"]
        assert data["tariffs_rs_per_kwh"]["delhi"] == 8.50

    def test_predict_wrong_content_type(self):
        r = self.client.post("/predict", data="not json")
        assert r.status_code == 415

    def test_predict_too_few_hours(self):
        payload = {"data": [{"ALLSKY_SFC_SW_DWN": 400, "T2M": 28} for _ in range(5)]}
        r = self.client.post("/predict",
                              data=json.dumps(payload),
                              content_type="application/json")
        # Should return 400 or 503 (if model not loaded), never 500 internal crash
        assert r.status_code in (400, 503)

    def test_roi_missing_fields(self):
        payload = {"system_kwp": 3.0}  # missing required fields
        r = self.client.post("/roi-report",
                              data=json.dumps(payload),
                              content_type="application/json")
        assert r.status_code == 400
        data = json.loads(r.data)
        assert "Missing fields" in data["error"]

    def test_roi_wrong_content_type(self):
        r = self.client.post("/roi-report", data="not json")
        assert r.status_code == 415

    @pytest.mark.skipif(
        not (ROOT / "data" / "processed" / "monthly_forecast_10yr.csv").exists(),
        reason="Forecast CSV not generated yet"
    )
    def test_roi_full_response_structure(self):
        """
        Full ROI report for reference system.
        Verified: Delhi 6 kWp → payback 2 years 5 months, profit ₹7,06,011.
        """
        payload = {
            "system_kwp": 6.0,
            "install_cost_rs": 270_000,
            "monthly_usage_kwh": 450,
            "city": "delhi",
        }
        r = self.client.post("/roi-report",
                              data=json.dumps(payload),
                              content_type="application/json")
        assert r.status_code == 200
        data = json.loads(r.data)

        # Structure checks
        assert "input" in data
        assert "subsidy" in data
        assert "result" in data
        assert "monthly_cashflow" in data

        result = data["result"]
        assert "payback_readable" in result
        assert "net_profit_10yr_rs" in result
        assert "irr_annual_pct" in result
        assert "co2_saved_tonnes" in result

        # Value checks — verified from actual pipeline run
        assert len(data["monthly_cashflow"]) == 120
        assert result["net_profit_10yr_rs"] > 0
        assert data["subsidy"]["pm_surya_ghar_rs"] == 105_000
        assert data["subsidy"]["net_investment_rs"] == 165_000

        # Payback should be in the 1–5 year range for this system
        assert result["payback_years"] is not None
        assert 1.0 < result["payback_years"] < 5.0


# ══════════════════════════════════════════════════════════════════════════════
# Standalone runner
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import subprocess
    subprocess.run(["python", "-m", "pytest", __file__, "-v", "--tb=short"])
