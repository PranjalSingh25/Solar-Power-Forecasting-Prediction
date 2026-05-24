import sys
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

class TestSubsidyCalculation:
    def setup_method(self):
        from solar_common import pm_surya_ghar_subsidy
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
        assert self.subsidy(4.0) == 78_000 + 9_000
        assert self.subsidy(6.0) == 78_000 + 27_000
        assert self.subsidy(10.0) == 78_000 + 63_000

    def test_above_10kwp_capped(self):
        assert self.subsidy(15.0) == self.subsidy(10.0)

class TestSystemSpec:
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
        assert spec.tariff == 7.00

    def test_net_cost_after_subsidy(self):
        spec = self.SystemSpec(3.0, 135_000, city="delhi")
        assert spec.subsidy == 78_000
        assert spec.net_cost == 135_000 - 78_000

    def test_net_cost_never_negative(self):
        spec = self.SystemSpec(1.0, 10_000, city="delhi")
        assert spec.net_cost >= 0

    def test_default_om_cost(self):
        spec = self.SystemSpec(4.0, 180_000, city="delhi")
        assert spec.annual_om_rs == 4.0 * 750

class TestROIEngine:
    FORECAST_CSV = ROOT / "data" / "processed" / "monthly_forecast_10yr.csv"

    def setup_method(self):
        from stage56_financial_roi import SystemSpec, compute_roi
        self.SystemSpec = SystemSpec
        self.compute_roi = compute_roi

    @pytest.mark.skipif(
        not FORECAST_CSV.exists(),
        reason="Forecast CSV not generated yet - run stage4_forecast_10yr.py first"
    )
    def test_delhi_6kwp_payback_range(self):
        spec = self.SystemSpec(6.0, 270_000, monthly_usage_kwh=450, city="delhi")
        result = self.compute_roi(spec)
        pb = result["payback_years"]
        assert pb is not None
        assert 1.5 < pb < 4.0, f"Expected ~2.4 years, got {pb}"

    @pytest.mark.skipif(not FORECAST_CSV.exists(), reason="Forecast CSV not generated yet")
    def test_10yr_profit_positive(self):
        spec = self.SystemSpec(6.0, 270_000, monthly_usage_kwh=450, city="delhi")
        result = self.compute_roi(spec)
        assert result["net_profit_10yr"] > 0

    @pytest.mark.skipif(not FORECAST_CSV.exists(), reason="Forecast CSV not generated yet")
    def test_irr_reasonable(self):
        spec = self.SystemSpec(6.0, 270_000, monthly_usage_kwh=450, city="delhi")
        result = self.compute_roi(spec)
        irr = result["irr_annual_pct"]
        assert irr is not None
        assert 10 < irr < 200, f"IRR out of expected range: {irr}"

    @pytest.mark.skipif(not FORECAST_CSV.exists(), reason="Forecast CSV not generated yet")
    def test_monthly_cashflow_length(self):
        spec = self.SystemSpec(6.0, 270_000, monthly_usage_kwh=450, city="delhi")
        result = self.compute_roi(spec)
        assert len(result["monthly_df"]) == 120

    @pytest.mark.skipif(not FORECAST_CSV.exists(), reason="Forecast CSV not generated yet")
    def test_cumulative_savings_monotonically_increases(self):
        spec = self.SystemSpec(6.0, 270_000, monthly_usage_kwh=450, city="delhi")
        result = self.compute_roi(spec)
        cumulative = result["monthly_df"]["cumulative_rs"].values
        assert all(cumulative[i] <= cumulative[i + 1] for i in range(len(cumulative) - 1))

    @pytest.mark.skipif(not FORECAST_CSV.exists(), reason="Forecast CSV not generated yet")
    def test_co2_savings_positive(self):
        spec = self.SystemSpec(6.0, 270_000, monthly_usage_kwh=450, city="delhi")
        result = self.compute_roi(spec)
        assert result["summary"]["co2_saved_tonnes"] > 0

class TestForecastData:
    FORECAST_CSV = ROOT / "data" / "processed" / "monthly_forecast_10yr.csv"

    @pytest.mark.skipif(not FORECAST_CSV.exists(), reason="Forecast CSV not generated yet")
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
        assert (df["kwh"] > 0).all()
        assert (df["kwh_degraded"] > 0).all()

    @pytest.mark.skipif(not FORECAST_CSV.exists(), reason="Forecast CSV not generated yet")
    def test_degradation_reduces_output(self):
        df = pd.read_csv(self.FORECAST_CSV)
        assert (df["kwh_degraded"] <= df["kwh"] + 0.01).all()

    @pytest.mark.skipif(not FORECAST_CSV.exists(), reason="Forecast CSV not generated yet")
    def test_year10_generation_less_than_year1(self):
        df = pd.read_csv(self.FORECAST_CSV)
        yr1_total  = df[df["year"] == 1]["kwh_degraded"].sum()
        yr10_total = df[df["year"] == 10]["kwh_degraded"].sum()
        assert yr10_total < yr1_total

    @pytest.mark.skipif(not FORECAST_CSV.exists(), reason="Forecast CSV not generated yet")
    def test_delhi_annual_kwh_in_realistic_range(self):
        df = pd.read_csv(self.FORECAST_CSV)
        yr1 = df[df["year"] == 1]["kwh_degraded"].sum()
        assert 5_000 < yr1 < 15_000, f"Annual generation out of range: {yr1:.0f} kWh"

class TestAPI:
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
        assert data["status"] in ("ok", "degraded")
        assert "forecast_ready" in data

    def test_tariffs_endpoint(self):
        r = self.client.get("/tariffs")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert "tariffs_rs_per_kwh" in data
        assert "delhi" in data["tariffs_rs_per_kwh"]
        assert data["tariffs_rs_per_kwh"]["delhi"] == 8.50

    def test_roi_missing_fields(self):
        payload = {"system_kwp": 3.0}
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

        assert "input" in data
        assert "subsidy" in data
        assert "result" in data
        assert "monthly_cashflow" in data

        result = data["result"]
        assert "payback_readable" in result
        assert "net_profit_10yr_rs" in result
        assert "irr_annual_pct" in result
        assert "co2_saved_tonnes" in result

        assert len(data["monthly_cashflow"]) == 120
        assert result["net_profit_10yr_rs"] > 0
        assert data["subsidy"]["pm_surya_ghar_rs"] == 105_000
        assert data["subsidy"]["net_investment_rs"] == 165_000
        assert result["payback_years"] is not None
        assert 1.0 < result["payback_years"] < 5.0

if __name__ == "__main__":
    import subprocess
    subprocess.run(["python", "-m", "pytest", __file__, "-v", "--tb=short"])
