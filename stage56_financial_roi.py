"""
Stage 5 + 6 — Financial Model & ROI Engine
India-specific solar investment analysis.
Inputs: system size, installation cost, household consumption, state tariff.
Outputs: month-by-month cash flow, exact payback month, 10-yr net profit, IRR.
"""

import numpy as np
import pandas as pd
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
PLOT_DIR     = BASE_DIR / "plots"
LOG_DIR      = BASE_DIR / "logs"
for d in [PLOT_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

FORECAST_CSV = DATA_DIR / "processed" / "monthly_forecast_10yr.csv"
ROI_CSV      = DATA_DIR / "processed" / "roi_analysis.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage56_roi.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("stage56")

# ─── State DISCOM Tariff Database (₹/kWh, residential, FY 2024-25) ─────────────
# Source: respective state electricity regulatory commission orders
STATE_TARIFFS = {
    "delhi":        8.50,
    "mumbai":       9.25,
    "bangalore":    7.10,
    "hyderabad":    6.30,
    "chennai":      5.80,
    "kolkata":      7.00,
    "pune":         9.00,
    "ahmedabad":    5.50,
    "jaipur":       6.80,
    "lucknow":      6.50,
    "chandigarh":   4.90,
    "bhopal":       7.20,
    "generic":      7.00,   # national average fallback
}

# PM Surya Ghar Yojana 2024 subsidy slabs (₹)
def pm_surya_ghar_subsidy(system_kwp: float) -> float:
    """Calculate PM Surya Ghar Muft Bijli Yojana subsidy (2024 scheme)."""
    if system_kwp <= 1:
        return 30_000
    elif system_kwp <= 2:
        return 60_000
    elif system_kwp <= 3:
        # ₹78,000 for the first 3 kW
        return 78_000
    else:
        # ₹78,000 for first 3 kWp + ₹9,000 per additional kWp up to 10 kWp
        extra_kwp = min(system_kwp - 3, 7)
        return 78_000 + extra_kwp * 9_000


@dataclass
class SystemSpec:
    """Everything needed to describe the solar installation."""
    system_kwp:      float          # installed peak capacity in kWp
    install_cost_rs: float          # total installation cost in ₹
    city:            str = "delhi"  # for tariff lookup
    monthly_usage_kwh: float = 300  # avg household monthly consumption
    net_metering_pct: float = 0.80  # fraction of excess exported at retail rate
    annual_om_rs:    float = None   # annual O&M cost; defaults to ₹750/kWp

    def __post_init__(self):
        if self.annual_om_rs is None:
            self.annual_om_rs = self.system_kwp * 750

    @property
    def tariff(self) -> float:
        return STATE_TARIFFS.get(self.city.lower().strip(), STATE_TARIFFS["generic"])

    @property
    def subsidy(self) -> float:
        return pm_surya_ghar_subsidy(self.system_kwp)

    @property
    def net_cost(self) -> float:
        return max(0, self.install_cost_rs - self.subsidy)


def compute_roi(spec: SystemSpec,
                forecast_csv: Path = FORECAST_CSV,
                tariff_escalation: float = 0.06,
                ) -> dict:
    """
    Core ROI computation.

    Returns a dict with:
      - monthly_df: month-by-month DataFrame
      - payback_month: int (1-indexed) or None if not within 10 yr
      - payback_years: float
      - net_profit_10yr: float (₹)
      - irr_10yr: float (%)
      - summary: dict of headline numbers
    """
    log.info("Loading 10-year forecast …")
    forecast = pd.read_csv(forecast_csv)
    log.info(f"  {len(forecast)} month-rows loaded")

    rows = []
    cumulative_savings = 0.0
    payback_month = None

    for _, row in forecast.iterrows():
        mo_idx = int(row["month_idx"])     # 1-indexed
        yr_num = int(row["year"])          # 1-indexed year
        kwh    = float(row["kwh_degraded"])

        # Tariff escalates annually
        tariff_this_year = spec.tariff * ((1 + tariff_escalation) ** (yr_num - 1))

        # Energy offset: how much of our consumption does solar cover?
        solar_covers = min(kwh, spec.monthly_usage_kwh)
        excess_kwh   = max(0, kwh - spec.monthly_usage_kwh)

        # Monthly savings
        grid_savings   = solar_covers * tariff_this_year
        export_credit  = excess_kwh * tariff_this_year * spec.net_metering_pct
        monthly_om     = spec.annual_om_rs / 12
        net_monthly    = grid_savings + export_credit - monthly_om

        cumulative_savings += net_monthly

        if payback_month is None and cumulative_savings >= spec.net_cost:
            payback_month = mo_idx

        rows.append({
            "month_idx":         mo_idx,
            "year":              yr_num,
            "month":             int(row["month"]),
            "kwh_generated":     round(kwh, 2),
            "tariff":            round(tariff_this_year, 4),
            "grid_savings_rs":   round(grid_savings, 2),
            "export_credit_rs":  round(export_credit, 2),
            "om_cost_rs":        round(monthly_om, 2),
            "net_monthly_rs":    round(net_monthly, 2),
            "cumulative_rs":     round(cumulative_savings, 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(ROI_CSV, index=False)
    log.info(f"ROI analysis saved → {ROI_CSV}")

    # ── Payback ────────────────────────────────────────────────────────────────
    if payback_month:
        payback_years = payback_month / 12
    else:
        payback_years = None
        log.warning("Payback not achieved within 10 years with current inputs.")

    # ── Net profit ─────────────────────────────────────────────────────────────
    net_profit_10yr = df["cumulative_rs"].iloc[-1] - spec.net_cost  \
                      if payback_month else df["cumulative_rs"].iloc[-1] - spec.net_cost

    # ── IRR (Internal Rate of Return) ─────────────────────────────────────────
    # Cash flows: -net_cost at month 0, then monthly net savings
    cashflows = [-spec.net_cost] + df["net_monthly_rs"].tolist()
    irr_monthly = _compute_irr(cashflows)
    irr_annual  = (1 + irr_monthly) ** 12 - 1 if irr_monthly else None

    # ── Summary ────────────────────────────────────────────────────────────────
    summary = {
        "system_kwp":           spec.system_kwp,
        "install_cost_rs":      spec.install_cost_rs,
        "pm_surya_ghar_subsidy": spec.subsidy,
        "net_cost_after_subsidy": spec.net_cost,
        "city":                 spec.city,
        "base_tariff":          spec.tariff,
        "monthly_usage_kwh":    spec.monthly_usage_kwh,
        "payback_month":        payback_month,
        "payback_years":        round(payback_years, 2) if payback_years else None,
        "net_profit_10yr_rs":   round(net_profit_10yr, 0),
        "irr_annual_pct":       round(irr_annual * 100, 2) if irr_annual else None,
        "total_10yr_kwh":       round(df["kwh_generated"].sum(), 0),
        "co2_saved_tonnes":     round(df["kwh_generated"].sum() * 0.82 / 1000, 2),
    }

    log.info("=== ROI Summary ===")
    for k, v in summary.items():
        log.info(f"  {k}: {v}")

    return {
        "monthly_df":       df,
        "payback_month":    payback_month,
        "payback_years":    payback_years,
        "net_profit_10yr":  net_profit_10yr,
        "irr_annual_pct":   irr_annual * 100 if irr_annual else None,
        "summary":          summary,
        "spec":             spec,
    }


def _compute_irr(cashflows: list, guess: float = 0.01,
                 max_iter: int = 1000, tol: float = 1e-6) -> Optional[float]:
    """Newton-Raphson IRR on monthly cashflows. Returns monthly IRR."""
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


def plot_roi(result: dict):
    """Generate the main ROI dashboard plot."""
    df   = result["monthly_df"]
    spec = result["spec"]
    summ = result["summary"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Solar ROI Analysis — {spec.system_kwp:.1f} kWp system in {spec.city.title()}",
        fontsize=14, fontweight="bold"
    )

    # 1. Monthly savings (bar)
    ax = axes[0, 0]
    colors = ["#2ecc71" if v > 0 else "#e74c3c"
              for v in df["net_monthly_rs"]]
    ax.bar(df["month_idx"], df["net_monthly_rs"], color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Monthly Net Savings (₹)")
    ax.set_xlabel("Month")
    ax.set_ylabel("₹")
    ax.grid(True, alpha=0.3)

    # 2. Cumulative cash flow — THE KEY CHART
    ax = axes[0, 1]
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(-spec.net_cost, color="red", linewidth=1.2,
               linestyle="--", label=f"Net investment ₹{spec.net_cost:,.0f}")
    ax.plot(df["month_idx"], df["cumulative_rs"] - spec.net_cost,
            color="#2980b9", linewidth=2, label="Cumulative profit/loss")
    ax.fill_between(df["month_idx"],
                    df["cumulative_rs"] - spec.net_cost, 0,
                    where=(df["cumulative_rs"] >= spec.net_cost),
                    alpha=0.3, color="green", label="Profit zone")
    ax.fill_between(df["month_idx"],
                    df["cumulative_rs"] - spec.net_cost, 0,
                    where=(df["cumulative_rs"] < spec.net_cost),
                    alpha=0.2, color="red", label="Loss zone")

    if result["payback_month"]:
        ax.axvline(result["payback_month"], color="orange", linewidth=2,
                   linestyle="-",
                   label=f"Breakeven: month {result['payback_month']}"
                         f" ({result['payback_years']:.1f} yrs)")
    ax.set_title("Cumulative Cash Flow (₹)")
    ax.set_xlabel("Month")
    ax.set_ylabel("₹")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Monthly kWh generation
    ax = axes[1, 0]
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    first_yr = df[df["year"] == 1]
    ax.bar(range(1, 13), first_yr["kwh_generated"],
           color="steelblue", alpha=0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, fontsize=9)
    ax.set_title("Monthly Generation — Year 1 (kWh)")
    ax.set_ylabel("kWh")
    ax.grid(True, alpha=0.3)

    # 4. Annual breakdown
    ax = axes[1, 1]
    annual = df.groupby("year").agg(
        generation=("kwh_generated", "sum"),
        savings=("grid_savings_rs", "sum"),
        export=("export_credit_rs", "sum"),
        om=("om_cost_rs", "sum"),
    )
    x = annual.index
    w = 0.35
    ax.bar(x - w/2, annual["savings"] + annual["export"],
           w, label="Gross savings", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, annual["om"],
           w, label="O&M cost", color="salmon", alpha=0.8)
    ax.set_title("Annual Savings vs O&M Cost (₹)")
    ax.set_xlabel("Year")
    ax.set_ylabel("₹")
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = PLOT_DIR / "roi_analysis.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"ROI plot saved → {out_path}")
    return out_path


def print_summary(result: dict):
    s = result["summary"]
    print("\n" + "═" * 55)
    print("  ☀  SOLAR ROI ANALYSIS REPORT")
    print("═" * 55)
    print(f"  System size:          {s['system_kwp']} kWp")
    print(f"  City:                 {s['city'].title()}")
    print(f"  Install cost:         ₹{s['install_cost_rs']:,.0f}")
    print(f"  PM Surya Ghar subsidy:₹{s['pm_surya_ghar_subsidy']:,.0f}")
    print(f"  Net investment:       ₹{s['net_cost_after_subsidy']:,.0f}")
    print("─" * 55)
    pb = result["payback_years"]
    if pb:
        yrs = int(pb)
        mos = round((pb - yrs) * 12)
        print(f"  ✅ PAYBACK PERIOD:    {yrs} years {mos} months")
    else:
        print(f"  ⚠  Payback >10 years with current inputs")
    print(f"  10-yr net profit:     ₹{s['net_profit_10yr_rs']:,.0f}")
    if s["irr_annual_pct"]:
        print(f"  Annual IRR:           {s['irr_annual_pct']:.1f}%")
    print("─" * 55)
    print(f"  Total generation:     {s['total_10yr_kwh']:,.0f} kWh")
    print(f"  CO₂ offset:           {s['co2_saved_tonnes']:.1f} tonnes")
    print("═" * 55)


if __name__ == "__main__":
    print("=== Stage 5+6: Financial Model & ROI Engine ===\n")

    # ── Get user inputs ────────────────────────────────────────────────────────
    print("System details (press Enter for defaults):\n")

    def ask(prompt, default, cast=float):
        val = input(f"  {prompt} [{default}]: ").strip()
        return cast(val) if val else default

    system_kwp      = ask("System size (kWp)",        3.0)
    install_cost    = ask("Installation cost (₹)",    135_000)
    monthly_usage   = ask("Monthly electricity use (kWh)", 300)
    city_input      = input(f"  City [{', '.join(list(STATE_TARIFFS.keys())[:6])}...] [delhi]: ").strip() or "delhi"

    print(f"\nAvailable cities: {', '.join(STATE_TARIFFS.keys())}")

    spec = SystemSpec(
        system_kwp=system_kwp,
        install_cost_rs=install_cost,
        monthly_usage_kwh=monthly_usage,
        city=city_input,
    )

    print(f"\n  Base tariff ({spec.city}): ₹{spec.tariff}/kWh")
    print(f"  PM Surya Ghar subsidy:   ₹{spec.subsidy:,.0f}")
    print(f"  Net investment:          ₹{spec.net_cost:,.0f}\n")

    result = compute_roi(spec)
    print_summary(result)

    plot_path = plot_roi(result)
    print(f"\n  Chart saved → {plot_path}")
    print(f"  Data saved  → {ROI_CSV}")
