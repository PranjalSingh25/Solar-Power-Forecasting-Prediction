<div align="center">

# ☀️ Solar ROI Predictor

**Stop guessing. Start knowing exactly when your solar investment pays off.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![pvlib](https://img.shields.io/badge/pvlib-Physics%20Engine-F7931E?style=for-the-badge)](https://pvlib-python.readthedocs.io)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Tests](https://img.shields.io/badge/Tests-34%20passed-22C55E?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-8B5CF6?style=for-the-badge)](LICENSE)

<br/>

*A physics-informed, ML-powered solar investment calculator built for the Indian residential market.*

<br/>

> *"With a 6 kWp system in Delhi at ₹2,70,000 installed cost —*
> ***you will break even in 2 years 5 months.***
> *10-year net profit: ₹7,06,011. Annual IRR: 54.6%."*
>
> *— actual output from this pipeline, New Delhi, verified June 2025*

</div>

---

## 📋 Table of Contents

- [The Problem](#-the-problem)
- [How It Works](#-how-it-works)
- [Why This Is Different](#-why-this-is-different)
- [Pipeline Architecture](#-pipeline-architecture)
- [Verified Results](#-verified-results)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Running the Pipeline](#-running-the-pipeline)
- [API Reference](#-api-reference)
- [Financial Assumptions](#-financial-model--assumptions)
- [Engineering Challenges](#-engineering-challenges-solved)
- [Known Limitations](#-known-limitations)
- [Roadmap](#-roadmap)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

---

## 🔴 The Problem

Every solar installer in India tells the same story: *"Solar is a long-term investment. You'll see returns in X years."*

That **X is always a guess** — and it's almost always wrong. Here's what it ignores:

**Shading.** A shadow from a water tank or neighbouring building can cut a panel's output by 30–50% during peak hours. Every rooftop is different. Most calculators assume yours is perfectly clear.

**Real weather patterns.** Delhi in January generates about 40% less solar energy than Delhi in June. Using an annual average hides the months where your panels barely cover your own consumption — which directly determines how fast your investment pays back.

**Tariff escalation.** Indian DISCOMs have raised electricity rates by 5–7% every year for the past decade. A calculator that freezes today's ₹8.50/kWh for 10 years will underestimate your lifetime savings by lakhs of rupees.

**Panel degradation.** Every solar panel loses ~0.5% of peak efficiency per year. After 10 years, your system produces 5% less than day one. No installer's brochure mentions this.

**Government subsidies.** PM Surya Ghar Muft Bijli Yojana (2024) offers ₹30,000–₹1,05,000 depending on system size. Most generic tools don't account for this at all.

The result: homeowners either get sold a system that doesn't make financial sense, or they talk themselves out of a genuinely profitable investment because the numbers were too vague to trust.

**This project gives you the real number — derived from your actual location, not a national average.**

---

## 💡 How It Works

A 7-stage pipeline takes GPS coordinates and your household details as input and produces a verified, month-by-month financial projection.

```
Coordinates + roof area + monthly bill + budget
                      ↓
     [ physics simulation + ML forecasting ]
                      ↓
   "Break even: 2 years 5 months. 10-yr profit: ₹7,06,011"
```

1. **Fetch real weather** — 5+ years of hourly satellite data from NASA POWER for your exact GPS location.
2. **Simulate your panels** — pvlib physics engine models your specific panel type, inverter, roof tilt, and orientation.
3. **Apply shading** — Sun position calculated every hour, crossed against your local horizon profile. Shaded hours are zeroed.
4. **Train a neural network** — PyTorch LSTM learns weather→power relationships from 8,700+ hours of data.
5. **Forecast 10 years** — Model predicts monthly generation for 120 months, with degradation applied year-by-year.
6. **Build the financial model** — kWh converted to ₹ using your DISCOM tariff, escalated 6%/yr, with subsidies and O&M deducted.
7. **Find the exact crossover** — Cumulative savings scanned month by month until it crosses your net investment cost.

---

## 🆚 Why This Is Different

| Feature | **This project** | Google Sunroof | PVWatts | MYSUN |
|---|:---:|:---:|:---:|:---:|
| India-specific | ✅ | ❌ US-only | ✅ | ✅ |
| Hourly weather data | ✅ | Estimated | ✅ | Unknown |
| ML generation forecast | ✅ | ❌ | ❌ | ❌ |
| Panel degradation modelled | ✅ | ❌ | ❌ | Unknown |
| Tariff escalation | ✅ | ❌ | ❌ | Unknown |
| PM Surya Ghar subsidy | ✅ | ❌ | ❌ | ✅ |
| IRR calculation | ✅ | ❌ | ❌ | ❌ |
| Exact payback month | ✅ | ❌ vague | ❌ | ❌ |
| Open source + free | ✅ | ❌ | ✅ | ❌ |

> Google Project Sunroof has been independently shown to misestimate payback periods by up to 4 years. Every assumption in this project is explicit, documented, and configurable.

---

## 🏗️ Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         USER INPUTS                                │
│  GPS coords · Roof area · Monthly bill · City · Budget             │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
┌─────────────────────┐        ┌────────────────────────────┐
│  STAGE 1            │        │  STAGE 2 (integrated)      │
│  Weather Fetch      │        │  Shadow / Horizon Analysis  │
│  NASA POWER API     │        │  pvlib sun position         │
│  5+ yrs · hourly   │        │  Local horizon profile      │
│  GHI·DNI·DHI·Temp  │        │  → shading_factor/hr        │
└──────────┬──────────┘        └──────────────┬──────────────┘
           └────────────────┬─────────────────┘
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — PV Physics Simulation                                   │
│  pvlib ModelChain · panel + inverter model · roof geometry         │
│  Weather → AC power (W/hr) · shading factor applied               │
└────────────────────────────┬───────────────────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — LSTM Training                                           │
│  PyTorch · 128 hidden · 2 layers · HuberLoss · AdamW              │
│  9 features incl. cyclical time encoding                           │
│  Test R² = 0.9839   RMSE = 181 W   MAE = 101 W                   │
└────────────────────────────┬───────────────────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — 10-Year Generation Forecast                             │
│  LSTM → monthly kWh · 0.5%/yr degradation curve                   │
│  Output: 120 months of predicted generation                        │
└────────────────────────────┬───────────────────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 5 — India Financial Model                                   │
│  13-city DISCOM tariffs · 6%/yr escalation                        │
│  PM Surya Ghar subsidy (auto) · net metering · O&M                │
└────────────────────────────┬───────────────────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 6 — ROI Engine                                              │
│  Month-by-month crossing point scan                                │
│  → Payback (years + months) · 10-yr profit · IRR                  │
└────────────────────────────┬───────────────────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 7 — REST API  (Flask)                                       │
│  POST /roi-report  →  full 10-year financial analysis JSON         │
│  POST /predict     →  next-hour AC power (W)                       │
│  GET  /tariffs     →  DISCOM rates + subsidy slabs                 │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Verified Results

All results produced by running the complete pipeline on **New Delhi (28.61°N, 77.20°E)**.
Test suite: **34 tests, 34 passed.**

---

### LSTM Model — Training Performance

| Metric | Value |
|---|---|
| R² (test set) | **0.9839** |
| RMSE | 181.3 W |
| MAE | 101.0 W |
| Split method | Chronological (no data leakage) |
| Training time | ~30 epochs, early stop |

![Training Results](plots/training_results.png)

*Left: Training vs validation loss over epochs. Right: Actual vs predicted scatter on the held-out test set.*

---

### 10-Year Generation Forecast — New Delhi, 6 kWp

| Year | Annual kWh (with degradation) |
|---|---|
| Year 1 | 9,093 kWh |
| Year 3 | 9,002 kWh |
| Year 5 | 8,913 kWh |
| Year 10 | 8,692 kWh |
| **10-yr total** | **88,913 kWh** |

![10-Year Forecast](plots/forecast_10yr.png)

*Top: Month-by-month generation for all 10 years (blue = no degradation, orange = with 0.5%/yr degradation). Bottom: Annual totals comparison.*

---

### ROI Analysis — 6 kWp, Delhi, ₹2,70,000 Installed

| Parameter | Value |
|---|---|
| PM Surya Ghar subsidy | ₹1,05,000 |
| Net investment after subsidy | ₹1,65,000 |
| Base DISCOM tariff (Delhi) | ₹8.50/kWh |
| Tariff escalation | 6%/year |
| **✅ Payback period** | **2 years 5 months** |
| **10-year net profit** | **₹7,06,011** |
| **Annual IRR** | **54.6%** |
| CO₂ offset over 10 years | 72.9 tonnes |

![ROI Analysis](plots/roi_analysis.png)

*Clockwise from top-left: monthly net savings, cumulative cash flow with breakeven line (orange), annual savings vs O&M, Year 1 monthly generation.*

---

### Test Suite

```
============================= test session starts ==============================
collected 34 items

tests/test_pipeline.py::TestSubsidyCalculation::test_below_1kwp         PASSED
tests/test_pipeline.py::TestSubsidyCalculation::test_1_to_2kwp          PASSED
tests/test_pipeline.py::TestSubsidyCalculation::test_2_to_3kwp          PASSED
tests/test_pipeline.py::TestSubsidyCalculation::test_above_3kwp         PASSED
tests/test_pipeline.py::TestSubsidyCalculation::test_above_10kwp_capped PASSED
tests/test_pipeline.py::TestSystemSpec::test_delhi_tariff                PASSED
tests/test_pipeline.py::TestSystemSpec::test_mumbai_tariff               PASSED
tests/test_pipeline.py::TestSystemSpec::test_unknown_city_fallback       PASSED
tests/test_pipeline.py::TestSystemSpec::test_net_cost_after_subsidy      PASSED
tests/test_pipeline.py::TestSystemSpec::test_net_cost_never_negative     PASSED
tests/test_pipeline.py::TestSystemSpec::test_default_om_cost             PASSED
tests/test_pipeline.py::TestROIEngine::test_delhi_6kwp_payback_range     PASSED
tests/test_pipeline.py::TestROIEngine::test_10yr_profit_positive         PASSED
tests/test_pipeline.py::TestROIEngine::test_irr_reasonable               PASSED
tests/test_pipeline.py::TestROIEngine::test_monthly_cashflow_length      PASSED
tests/test_pipeline.py::TestROIEngine::test_cumulative_savings_*         PASSED
tests/test_pipeline.py::TestROIEngine::test_co2_savings_positive         PASSED
tests/test_pipeline.py::TestLSTMInference::test_model_loads_*            PASSED
tests/test_pipeline.py::TestLSTMInference::test_prediction_non_negative  PASSED
tests/test_pipeline.py::TestLSTMInference::test_model_r2_above_threshold PASSED
tests/test_pipeline.py::TestForecastData::test_forecast_has_120_rows     PASSED
tests/test_pipeline.py::TestForecastData::test_forecast_columns_present  PASSED
tests/test_pipeline.py::TestForecastData::test_kwh_values_positive       PASSED
tests/test_pipeline.py::TestForecastData::test_degradation_reduces_*     PASSED
tests/test_pipeline.py::TestForecastData::test_year10_less_than_year1    PASSED
tests/test_pipeline.py::TestForecastData::test_delhi_annual_kwh_range    PASSED
tests/test_pipeline.py::TestAPI::test_root_returns_200                   PASSED
tests/test_pipeline.py::TestAPI::test_health_endpoint                    PASSED
tests/test_pipeline.py::TestAPI::test_tariffs_endpoint                   PASSED
tests/test_pipeline.py::TestAPI::test_predict_wrong_content_type         PASSED
tests/test_pipeline.py::TestAPI::test_predict_too_few_hours              PASSED
tests/test_pipeline.py::TestAPI::test_roi_missing_fields                 PASSED
tests/test_pipeline.py::TestAPI::test_roi_wrong_content_type             PASSED
tests/test_pipeline.py::TestAPI::test_roi_full_response_structure        PASSED

==================== 34 passed in 69.62s ====================
```

Run the tests yourself:
```bash
pip install pytest
python -m pytest tests/ -v
```

---

## 🛠️ Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Weather data | NASA POWER API | Free, global, satellite-validated hourly data since 2000 |
| PV physics | pvlib-python | Certified simulation library used by NREL and IEA |
| Deep learning | PyTorch LSTM | Captures time-of-day and seasonal dependencies |
| Preprocessing | scikit-learn + joblib | Scaler state preserved between training and inference |
| API | Flask | Lightweight, self-hostable |
| Data | pandas + numpy | |
| Timezone | timezonefinder | Auto-detects timezone from GPS |
| Visualisation | matplotlib | |

---

## ⚙️ Getting Started

**Prerequisites:** Python 3.8+

```bash
# Clone
git clone https://github.com/PranjalSingh25/Solar-Power-Forecasting-Prediction
cd Solar-Power-Forecasting-Prediction

# Virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

Run stages in order. Each stage writes a file the next stage reads.

### Stage 1 — Fetch Weather

```bash
python stage1_fetch_weather.py
# Prompts: latitude, longitude, start year, end year
# Output:  data/nasa_power_hourly_raw.csv
```

Fetch at least 3 years for reliable seasonal patterns. The script chunks large requests year-by-year automatically.

---

### Stage 2 — PV Simulation + Shadow Analysis

Edit the system config at the top of `stage2_simulate_pv.py`:

```python
LATITUDE           = 28.6139    # your location
LONGITUDE          = 77.2090
SURFACE_TILT       = 28         # degrees from horizontal
SURFACE_AZIMUTH    = 180        # 180 = south-facing
MODULES_PER_STRING = 10
STRINGS_PER_INV    = 2
HORIZON_ELEVATIONS = [5.0] * 36 # degrees of obstruction per azimuth direction
```

```bash
python stage2_simulate_pv.py
# Output: data/processed/weather_and_simulated_hourly_power.csv
#         (includes simulated_ac_power_W, shading_factor, ac_power_shaded_W)
```

---

### Stage 3 — Train the LSTM

```bash
python stage3_train_lstm.py

# Custom hyperparameters:
python stage3_train_lstm.py --epochs 100 --seq_len 48 --hidden 256 --lr 0.0005
```

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 60 | Max training epochs |
| `--seq_len` | 24 | Input window (hours) |
| `--hidden` | 128 | LSTM hidden units |
| `--layers` | 2 | LSTM stacked layers |
| `--lr` | 0.001 | Learning rate |
| `--patience` | 12 | Early stopping patience |

```
# Output:
models/best_lstm_model_hourly.pth
models/feature_scaler_hourly.joblib
models/target_scaler_hourly.joblib
plots/training_results.png
```

---

### Stage 4 — 10-Year Forecast

```bash
python stage4_forecast_10yr.py
# Output: data/processed/monthly_forecast_10yr.csv
#         plots/forecast_10yr.png
```

---

### Stage 5+6 — Financial Model & ROI

```bash
python stage56_financial_roi.py
# Prompts: system size, install cost, monthly usage, city
# Output:  data/processed/roi_analysis.csv
#          plots/roi_analysis.png
```

---

### Stage 7 — Start the API

```bash
python app.py
# Running at http://127.0.0.1:5001
```

---

## 🔌 API Reference

### `GET /health`

```json
{ "status": "ok", "model": true, "forecast_ready": true, "device": "cpu" }
```

### `GET /tariffs`

Returns all 13 supported cities with DISCOM tariffs and PM Surya Ghar subsidy slabs.

### `POST /predict`

Next-hour AC power from 24 hours of weather data.

```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [ ... 24 hourly weather dicts ... ]}'
```

```json
{ "predicted_power_W": 1847.32, "predicted_power_kW": 1.8473, "unit": "Watts" }
```

### `POST /roi-report` ⭐

Full 10-year financial analysis.

```bash
curl -X POST http://127.0.0.1:5001/roi-report \
  -H "Content-Type: application/json" \
  -d '{
    "system_kwp":         6.0,
    "install_cost_rs":    270000,
    "monthly_usage_kwh":  450,
    "city":               "delhi",
    "net_metering_pct":   0.80,
    "tariff_escalation":  0.06
  }'
```

```json
{
  "input":   { "system_kwp": 6.0, "city": "delhi", "base_tariff": 8.5 },
  "subsidy": { "pm_surya_ghar_rs": 105000, "net_investment_rs": 165000 },
  "result":  {
    "payback_readable":   "2 years 5 months",
    "net_profit_10yr_rs": 706011,
    "irr_annual_pct":     54.56,
    "total_10yr_kwh":     88913,
    "co2_saved_tonnes":   72.91
  },
  "monthly_cashflow": [ ... 120 rows ... ]
}
```

**Supported cities:** `delhi` · `mumbai` · `bangalore` · `hyderabad` · `chennai` · `kolkata` · `pune` · `ahmedabad` · `jaipur` · `lucknow` · `chandigarh` · `bhopal` · `generic`

---

## 💰 Financial Model — Assumptions

Every parameter is sourced, documented, and configurable.

| Parameter | Default | Source |
|---|---|---|
| Panel degradation | 0.5%/year | IEC 61215 |
| Tariff escalation | 6%/year | Historical DISCOM average 2015–2024 |
| System cost | ₹45,000/kWp | 2024 Indian residential market |
| O&M cost | ₹750/kWp/year | Conservative estimate |
| Net metering credit | 80% of retail tariff | Average across DISCOMs |
| Subsidy ≤1 kWp | ₹30,000 | MNRE Feb 2024 |
| Subsidy ≤2 kWp | ₹60,000 | MNRE Feb 2024 |
| Subsidy ≤3 kWp | ₹78,000 | MNRE Feb 2024 |
| Subsidy 3–10 kWp | ₹78,000 + ₹9,000/kWp | MNRE Feb 2024 |
| CO₂ factor | 0.82 kg/kWh | CEA India 2023 |

---

## 🔧 Engineering Challenges Solved

**Unit mismatch — daily vs hourly data.** The first version fetched daily averaged irradiance. pvlib's `ModelChain` requires instantaneous hourly W/m² values — feeding daily averages produced uniform output across all 24 hours, including night. Fix: refactored to fetch hourly data from NASA POWER.

**Missing irradiance components.** pvlib requires GHI, DNI, and DHI independently. Early code fetched only GHI and relied on internal decomposition, which introduces large errors under Indian monsoon cloud conditions. Fix: all three fetched directly from NASA POWER.

**The scaler trap.** Post-training predictions were wildly wrong despite good loss curves. Cause: a new `MinMaxScaler` was re-fitted on inference data, giving different bounds than the training scaler. Fix: save both scalers with `joblib.dump()` immediately after fitting. Load — never refit — at inference time.

**Uninitialized ModelChain.** Python runtime errors accessing `mc.results` because `mc` was declared inside a `try` block and referenced outside. Fix: initialize `mc = None` before the block, add `None` guard before accessing results.

**Poor seasonal accuracy.** LSTM predicted similar output for January and June noon despite ~40% real-world difference. Fix: added cyclical time features (`hour_sin`, `hour_cos`, `month_sin`, `month_cos`) so the model understands both time-of-day and time-of-year as continuous signals.

---

## ⚠️ Known Limitations

**Trained on simulated data.** R² = 0.9839 measures replication of pvlib's physics model — not a real solar meter. Real-world factors (dust soiling, partial shading, inverter faults) are not modelled. Expect actual generation to be 5–15% below projections.

**One year of training data.** The current model was trained on 2016 weather for New Delhi. Training on 5+ years substantially improves monsoon season accuracy.

**Approximate horizon shading.** Current implementation uses a uniform 5° obstruction angle. Real urban horizons are irregular. Precise profiles require a site survey or LiDAR data.

**India-specific financial model.** The physics pipeline works globally. The financial model (tariffs, subsidies, net metering) is built for India and would need adaptation for other markets.

---

## 🗺️ Roadmap

| Feature | Priority |
|---|---|
| Real measured generation data (inverter APIs) | High |
| Residual modelling (sim-to-reality gap) | High |
| OpenWeatherMap 5-day live forecast | Medium |
| Precise shading via LiDAR / Google Solar API | Medium |
| Expanded city + rural DISCOM database | Medium |
| Docker + cloud deployment | Low |
| Streamlit / React frontend | Low |

---

## 📁 Project Structure

```
Solar-Power-Forecasting-Prediction/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── stage1_fetch_weather.py          # NASA POWER hourly data fetcher
├── stage2_simulate_pv.py            # pvlib simulation + shadow analysis
├── stage3_train_lstm.py             # LSTM training + evaluation
├── stage4_forecast_10yr.py          # 10-year monthly generation forecast
├── stage56_financial_roi.py         # Financial model + ROI engine
├── app.py                           # Flask REST API
│
├── tests/
│   └── test_pipeline.py             # 34-test pytest suite
│
├── plots/
│   ├── training_results.png         # Loss curve + prediction scatter
│   ├── forecast_10yr.png            # Monthly + annual generation charts
│   └── roi_analysis.png             # 4-panel ROI dashboard
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── logs/
│   └── .gitkeep
│
└── config/
```

> Model weights, raw data, and processed CSVs are excluded from version control (see `.gitignore`). Run the pipeline stages in order to regenerate them.

---

## 🧹 Repo Clean-up Guide

Coming from an older version of this project? Remove committed junk before pushing the new structure:

```bash
# Create .gitignore first, then remove tracked files that should be ignored
git rm -r --cached venv/
git rm -r --cached __pycache__/
git rm --cached app_log.txt solar_data_processing.log
git rm --cached environment.yml
git rm --cached solar_data_pipeline_2.py simulate_pv_power.py train_lstm.py

# Commit the cleanup
git add .gitignore requirements.txt
git commit -m "chore: clean repo structure, add gitignore and requirements"
```

---

## 🤝 Contributing

Contributions are welcome. The highest-impact areas:

- **Real generation data** — smart inverter data or DISCOM records would be the single biggest accuracy improvement.
- **More city tariffs** — accurate DISCOM rates for Tier-2 cities and rural feeders.
- **Shading precision** — Open-Meteo horizon data or Google Solar API integration.

Please open an issue before starting significant work.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Pranjal Singh**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/pranjal-singh-265937286/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/PranjalSingh25)

---

<div align="center">

*Built to answer one question honestly:*

**"Is solar actually worth it for me — and exactly when will I get my money back?"**

</div>
