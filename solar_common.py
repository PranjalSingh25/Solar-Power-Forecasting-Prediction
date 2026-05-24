STATE_TARIFFS = {
    "delhi": 8.50, "mumbai": 9.25, "bangalore": 7.10,
    "hyderabad": 6.30, "chennai": 5.80, "kolkata": 7.00,
    "pune": 9.00, "ahmedabad": 5.50, "jaipur": 6.80,
    "lucknow": 6.50, "chandigarh": 4.90, "bhopal": 7.20,
    "generic": 7.00,
}

def pm_surya_ghar_subsidy(kwp):
    if kwp <= 1:
        return 30_000
    elif kwp <= 2:
        return 60_000
    elif kwp <= 3:
        return 78_000
    else:
        extra_kwp = min(kwp - 3, 7)
        return 78_000 + extra_kwp * 9_000

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
