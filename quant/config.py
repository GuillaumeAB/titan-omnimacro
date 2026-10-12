"""
Global configuration for the quant engine.

Every number here is an *assumption* you can change. Values that come from
index-provider methodologies change over time — verify them against the
current S&P / Nasdaq methodology documents before relying on them.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
OUTPUT_DIR = ROOT / "output"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data
PRICE_START = "2005-01-01"
CACHE_MAX_AGE_HOURS = 20
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
}

# ---------------------------------------------------------------- model
WEEKS_PER_YEAR = 52
MIN_HISTORY_WEEKS = 104          # assets with less history are excluded from the factor fit
ESTIMATION_WINDOW_WEEKS = 52 * 15  # rolling window used for betas / factors
N_STAT_FACTORS = 6               # PCA factors extracted from macro-model residuals
N_REGIMES = 3
HMM_RESTARTS = 4
ALPHA_SHRINKAGE = 0.15           # keep only 15% of historical alpha (the rest is noise)

# Forward-looking factor premia (annual, arithmetic, in excess of cash).
# Only the global equity factor carries a meaningful premium by default.
# Change these to express your own long-run views.
FACTOR_PREMIA_ANNUAL = {
    "MKT": 0.040,      # global equity risk premium
    "USD": 0.0,
    "RATES": 0.0,      # change in 10Y yield (pp) — no drift
    "OIL": 0.0,
    "COPPER": 0.0,
    "GOLD": 0.0,
    "CREDIT": 0.010,   # HY-minus-Treasury carry
    "EM": 0.0,
}

HORIZONS_WEEKS = {"3M": 13, "6M": 26, "1Y": 52, "2Y": 104, "3Y": 156, "5Y": 260}

# Observable macro factors -> Yahoo tickers used to build them (weekly).
#   MKT    = log return of ACWI
#   USD    = log return of DX-Y.NYB (dollar index)
#   RATES  = weekly CHANGE of ^TNX (10Y yield, in percentage points)
#   OIL    = log return of BZ=F (Brent; WTI CL=F went negative in 2020)
#   COPPER = log return of HG=F
#   GOLD   = log return of GC=F
#   CREDIT = log return HYG minus log return IEF
#   EM     = log return EEM minus log return ACWI
FACTOR_PROXIES = {
    "MKT": ["ACWI"],
    "USD": ["DX-Y.NYB"],
    "RATES": ["^TNX"],
    "OIL": ["BZ=F"],
    "COPPER": ["HG=F"],
    "GOLD": ["GC=F"],
    "CREDIT": ["HYG", "IEF"],
    "EM": ["EEM", "ACWI"],
}
FACTOR_PROXY_TICKERS = sorted({t for v in FACTOR_PROXIES.values() for t in v})
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# ---------------------------------------------------------------- index rules
# S&P 500 eligibility (S&P U.S. Indices Methodology). Thresholds are reviewed
# periodically by S&P DJI — VERIFY the current value before use.
SP500_MIN_MCAP_USD = 22.7e9
SP500_MIN_FLOAT = 0.10
SP500_IPO_SEASONING_MONTHS = 12

# Nasdaq-100 annual reconstitution (December). Ranking is based on market cap
# at the end of October. Members ranked 101–125 may be retained if they were in
# the top 100 at the previous reconstitution. VERIFY the current methodology
# (seasoning / fast-entry rules have been revised in the past).
NDX_SIZE = 100
NDX_BUFFER_RANK = 125
NDX_SEASONING_MONTHS = 3
