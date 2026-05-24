import logging
import sys
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, make_response
from pathlib import Path

from solar_common import STATE_TARIFFS, pm_surya_ghar_subsidy, _compute_irr

BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
LOG_DIR     = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

FORECAST_CSV = DATA_DIR / "processed" / "monthly_forecast_10yr.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "api.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("api")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    accept = request.headers.get("Accept", "")
    wants_html = "text/html" in accept and ("application/json" not in accept or accept.index("text/html") < accept.index("application/json"))
    if not wants_html:
        return jsonify({
            "service": "Solar ROI Prediction API",
            "endpoints": {
                "GET  /health":     "Service health check",
                "POST /roi-report": "Full 10-year ROI analysis",
                "GET  /tariffs":    "Supported cities and tariffs",
            },
            "forecast_ready": FORECAST_CSV.exists(),
        })

    try:
        result = _run_roi({
            "system_kwp": 6.0,
            "install_cost_rs": 270000,
            "monthly_usage_kwh": 450,
            "city": "delhi",
        })
    except Exception:
        return "<html><body><h1>Solar ROI API</h1><p>Forecast data not ready. Run stage4_forecast_10yr.py first.</p></body></html>"

    inp  = result["input"]
    sub  = result["subsidy"]
    res  = result["result"]
    pb   = res["payback_years"]
    yrs  = int(pb)
    mos  = round((pb - yrs) * 12)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Solar ROI Predictor</title>
<style>
  body {{ font-family: Helvetica, Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; color: #333; }}
  h1 {{ text-align: center; color: #1a5276; }}
  .report {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; }}
  .row {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #eee; }}
  .row:last-child {{ border: none; }}
  .label {{ color: #555; }}
  .value {{ font-weight: bold; }}
  .highlight {{ background: #d4edda; border-radius: 6px; padding: 10px 14px; margin: 12px 0; }}
  .highlight .row {{ border: none; }}
  .endpoints {{ margin-top: 24px; font-size: 0.85em; color: #666; }}
  code {{ background: #e9ecef; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Solar ROI Predictor</h1>
<div class="report">
  <div class="row"><span class="label">System</span><span class="value">{inp['system_kwp']} kWp</span></div>
  <div class="row"><span class="label">City</span><span class="value">{inp['city'].title()}</span></div>
  <div class="row"><span class="label">Install cost</span><span class="value">Rs {inp['install_cost_rs']:,}</span></div>
  <div class="row"><span class="label">PM Surya Ghar subsidy</span><span class="value" style="color:#27ae60;">Rs {sub['pm_surya_ghar_rs']:,}</span></div>
  <div class="row"><span class="label">Net investment</span><span class="value">Rs {sub['net_investment_rs']:,}</span></div>

  <div class="highlight">
    <div class="row"><span class="label">Payback period</span><span class="value" style="color:#e67e22;">{yrs} years {mos} months</span></div>
    <div class="row"><span class="label">10-year net profit</span><span class="value" style="color:#27ae60;">Rs {res['net_profit_10yr_rs']:,}</span></div>
    <div class="row"><span class="label">Annual IRR</span><span class="value" style="color:#2980b9;">{res['irr_annual_pct']:.1f}%</span></div>
  </div>

  <div class="row"><span class="label">Total generation</span><span class="value">{res['total_10yr_kwh']:,} kWh</span></div>
  <div class="row"><span class="label">CO2 offset</span><span class="value">{res['co2_saved_tonnes']:.1f} tonnes</span></div>
</div>

<div class="endpoints">
  <p>API endpoints:</p>
  <p><code>POST /roi-report</code> — custom scenario</p>
  <p><code>GET /tariffs</code> — DISCOM rates</p>
  <p><code>GET /health</code> — status</p>
</div>
</body>
</html>"""
    return make_response(html, 200, {"Content-Type": "text/html; charset=utf-8"})

@app.route("/health", methods=["GET"])
def health():
    ok = FORECAST_CSV.exists()
    return jsonify({
        "status": "ok" if ok else "degraded",
        "forecast_ready": FORECAST_CSV.exists(),
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

@app.route("/roi-report", methods=["POST"])
def roi_report():
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

def _run_roi(body):
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

    if not FORECAST_CSV.exists():
        raise FileNotFoundError(
            "monthly_forecast_10yr.csv not found. Run stage4_forecast_10yr.py first."
        )
    forecast = pd.read_csv(FORECAST_CSV)

    monthly_rows = []
    cumulative   = 0.0
    payback_month = None
    cashflows    = [-net_cost]

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

def _print_report(result):
    inp  = result["input"]
    sub  = result["subsidy"]
    res  = result["result"]
    pb   = res["payback_years"]
    yrs  = int(pb)
    mos  = round((pb - yrs) * 12)

    print("\n" + "=" * 55)
    print("  SOLAR ROI ANALYSIS REPORT")
    print("=" * 55)
    print(f"  System size:          {inp['system_kwp']} kWp")
    print(f"  City:                 {inp['city'].title()}")
    print(f"  Install cost:         Rs{inp['install_cost_rs']:,.0f}")
    print(f"  PM Surya Ghar subsidy:Rs{sub['pm_surya_ghar_rs']:,.0f}")
    print(f"  Net investment:       Rs{sub['net_investment_rs']:,.0f}")
    print("-" * 55)
    print(f"  PAYBACK PERIOD:       {yrs} years {mos} months")
    print(f"  10-yr net profit:     Rs{res['net_profit_10yr_rs']:,.0f}")
    print(f"  Annual IRR:           {res['irr_annual_pct']:.1f}%")
    print("-" * 55)
    print(f"  Total generation:     {res['total_10yr_kwh']:,.0f} kWh")
    print(f"  CO2 offset:           {res['co2_saved_tonnes']:.1f} tonnes")
    print("=" * 55)

if __name__ == "__main__":
    print("=== Solar ROI Prediction API ===")
    try:
        result = _run_roi({
            "system_kwp": 6.0,
            "install_cost_rs": 270000,
            "monthly_usage_kwh": 450,
            "city": "delhi",
        })
        _print_report(result)
    except Exception as e:
        log.warning(f"Default report skipped: {e}")
    host = "127.0.0.1"
    port = 5001
    log.info(f"Starting Solar ROI API on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
