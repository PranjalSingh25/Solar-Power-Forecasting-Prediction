<div align="center">

# Solar ROI Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![pvlib](https://img.shields.io/badge/pvlib-Physics%20Engine-F7931E?style=for-the-badge)](https://pvlib-python.readthedocs.io)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Tests](https://img.shields.io/badge/Tests-29%20passed-22C55E?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-8B5CF6?style=for-the-badge)](LICENSE)

A solar investment calculator for the Indian residential market.

> With a 6 kWp system in Delhi at Rs 2,70,000 installed cost,
> break even is 2 years 5 months.
> 10-year net profit: Rs 6,83,732. Annual IRR: 53.0%.

</div>

## The Problem

Solar installers in India will tell you its a long term investment and you will see returns in X years. That X is usually a guess. Heres why:

**Shading.** A shadow from a water tank or a neighbouring building can cut panel output by 30-50% during peak hours. Most calculators assume a perfectly clear rooftop.

**Weather patterns.** Delhi in January generates about 40% less solar energy than Delhi in June. Annual averages hide the months where your panels barely cover your own consumption, and that directly determines how fast your investment pays back.

**Tariff escalation.** Indian DISCOMs have raised electricity rates by 5-7% every year for the past decade. A calculator that freezes todays rate for 10 years underestimates your lifetime savings by lakhs of rupees.

**Panel degradation.** Solar panels lose about 0.5% of peak efficiency per year. After 10 years your system produces 5% less than day one. Most brochures dont mention this.

**Government subsidies.** The PM Surya Ghar Muft Bijli Yojana (2024) offers Rs 30,000 to Rs 1,05,000 depending on system size. Generic tools dont account for this.

This project gives you the real number derived from your actual location, not a national average.

## How It Works

```
GPS coordinates + monthly bill + budget
                    |
     [ pvlib physics simulation ]
                    |
   "Break even: 2 years 5 months. 10-yr profit: Rs 7,06,011"
```

1. Fetch 5+ years of hourly satellite weather data from NASA POWER for your GPS location.
2. pvlib simulates hourly AC power using your panel type, inverter, roof tilt, and orientation, including horizon-based shading.
3. Hourly output is grouped by month and averaged across all years to build a 12-month climatological profile.
4. That profile is tiled across 10 years with 0.5%/yr panel degradation.
5. Financial model converts kWh to rupees using DISCOM tariffs (escalated 6%/yr), applies PM Surya Ghar subsidy, subtracts O&M, and scans month-by-month for the payback crossover.

## Pipeline

```
STAGE 1: Weather Fetch
  NASA POWER API (hourly, multi-year)
  GHI, DNI, DHI, temperature, wind speed

STAGE 2: PV Physics Simulation
  pvlib ModelChain (panel model + inverter + shading factor)
  Weather -> hourly AC power (W)

STAGE 3: 10-Year Forecast
  Monthly average across all available years
  Tile 12-month pattern 10x with 0.5%/yr degradation
  Output: 120 months of projected generation (kWh)

STAGE 4: Financial Model & ROI
  13-city DISCOM tariffs, 6%/yr escalation
  PM Surya Ghar subsidy, net metering, O&M
  Month-by-month crossover scan
  -> Payback period, 10-yr profit, IRR, CO2 offset
```

## Results

All results from running the pipeline on New Delhi (28.61N, 77.20E).

### 10-Year Generation Forecast - New Delhi, 6 kWp

| Year | Annual kWh (with degradation) |
|------|------------------------------|
| Year 1 | 8,838 kWh |
| Year 3 | 8,750 kWh |
| Year 5 | 8,663 kWh |
| Year 10 | 8,448 kWh |
| 10-yr total | 86,421 kWh |

![10-Year Forecast](plots/forecast_10yr.png)

### ROI Analysis - 6 kWp, Delhi, Rs 2,70,000 Installed

| Parameter | Value |
|-----------|-------|
| PM Surya Ghar subsidy | Rs 1,05,000 |
| Net investment after subsidy | Rs 1,65,000 |
| Base DISCOM tariff (Delhi) | Rs 8.50/kWh |
| Tariff escalation | 6%/year |
| Payback period | 2 years 5 months |
| 10-year net profit | Rs 6,83,732 |
| Annual IRR | 53.0% |
| CO2 offset over 10 years | 70.9 tonnes |

![ROI Analysis](plots/roi_analysis.png)

## Tech Stack

| Layer | Tool |
|-------|------|
| Weather data | NASA POWER API |
| PV physics | pvlib-python |
| API | Flask |
| Data | pandas + numpy |
| Timezone | timezonefinder |
| Visualisation | matplotlib |

## Getting Started

```bash
git clone https://github.com/PranjalSingh25/Solar-Power-Forecasting-Prediction
cd Solar-Power-Forecasting-Prediction
pip install -r requirements.txt
```

## Running the Pipeline

### Stage 1 - Fetch Weather
```bash
python stage1_fetch_weather.py
```
Prompts for latitude, longitude, start year, end year. Output: `data/nasa_power_hourly_raw.csv`

### Stage 2 - PV Simulation + Shadow Analysis
Edit the config at the top of `stage2_simulate_pv.py` to match your hardware, then:
```bash
python stage2_simulate_pv.py
```
Output: `data/processed/weather_and_simulated_hourly_power.csv`

### Stage 3 - 10-Year Forecast
```bash
python stage4_forecast_10yr.py
```
Output: `data/processed/monthly_forecast_10yr.csv`, `plots/forecast_10yr.png`

### Stage 4 - Financial Model & ROI
```bash
python stage56_financial_roi.py
```
Output: `data/processed/roi_analysis.csv`, `plots/roi_analysis.png`

### Start the API
```bash
python app.py
# Running at http://127.0.0.1:5001
```

## API Reference

### GET /health
```json
{ "status": "ok", "forecast_ready": true }
```

### GET /tariffs
Returns all supported cities with DISCOM tariffs and PM Surya Ghar subsidy slabs.

### POST /roi-report
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
    "net_profit_10yr_rs": 683732,
    "irr_annual_pct":     53.04,
    "total_10yr_kwh":     86421,
    "co2_saved_tonnes":   70.87
  },
  "monthly_cashflow": [ ... 120 rows ... ]
}
```

Supported cities: `delhi`, `mumbai`, `bangalore`, `hyderabad`, `chennai`, `kolkata`, `pune`, `ahmedabad`, `jaipur`, `lucknow`, `chandigarh`, `bhopal`, `generic`

## Financial Model

| Parameter | Default | Source |
|-----------|---------|--------|
| Panel degradation | 0.5%/year | IEC 61215 |
| Tariff escalation | 6%/year | Historical DISCOM average 2015-2024 |
| System cost | Rs 45,000/kWp | 2024 Indian residential market |
| O&M cost | Rs 750/kWp/year | Conservative estimate |
| Net metering credit | 80% of retail tariff | Average across DISCOMs |
| Subsidy <=1 kWp | Rs 30,000 | MNRE Feb 2024 |
| Subsidy <=2 kWp | Rs 60,000 | MNRE Feb 2024 |
| Subsidy <=3 kWp | Rs 78,000 | MNRE Feb 2024 |
| Subsidy 3-10 kWp | Rs 78,000 + Rs 9,000/kWp | MNRE Feb 2024 |
| CO2 factor | 0.82 kg/kWh | CEA India 2023 |

## Known Limitations

**Simulated, not measured.** Generation is computed by pvlib physics models, not from a real solar meter. Real-world factors (dust soiling, partial shading, inverter faults) are not modelled. Actual generation may be 5-15% below projections.

**Approximate horizon shading.** Uses a uniform 5 degree obstruction angle. Real urban horizons are irregular. Precise profiles require a site survey or LiDAR data.

**India-specific financial model.** The physics pipeline works globally. The financial model (tariffs, subsidies, net metering) is built for India and would need adaptation for other markets.

## Project Structure

```
Solar-Power-Forecasting-Prediction/
+-- stage1_fetch_weather.py
+-- stage2_simulate_pv.py
+-- stage4_forecast_10yr.py
+-- stage56_financial_roi.py
+-- solar_common.py
+-- app.py
+-- tests/
|   +-- test_pipeline.py
+-- plots/
|   +-- forecast_10yr.png
|   +-- roi_analysis.png
+-- data/
+-- config/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Pranjal Singh**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/pranjal-singh-265937286/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/PranjalSingh25)

<div align="center">

*Is solar actually worth it for me, and exactly when will I get my money back?*

</div>
