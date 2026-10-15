"""
Global configuration.

Every number in this file is an *assumption*. Nothing here is estimated from data;
change them to express your own views. Index-methodology constants drift over time —
verify them against the current S&P DJI / Nasdaq documents before relying on them.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("TITAN_CACHE_DIR", ROOT / ".cache"))
OUTPUT_DIR = Path(os.environ.get("TITAN_OUTPUT_DIR", ROOT / "output"))

# ---------------------------------------------------------------- data
PRICE_START = "2005-01-01"
CACHE_MAX_AGE_HOURS = 20
HTTP_TIMEOUT = 20
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
}

# ---------------------------------------------------------------- calendar
WEEKS_PER_YEAR = 52
WEEK_RULE = "W-FRI"
HORIZONS_WEEKS = {"3M": 13, "6M": 26, "1Y": 52, "2Y": 104, "3Y": 156, "5Y": 260}
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

# ---------------------------------------------------------------- factor model
# Observable macro factors (weekly):
#   MKT    log return of ACWI (total return, dividends included by auto-adjust)
#   USD    log return of the dollar index DX-Y.NYB
#   RATES  weekly CHANGE of ^TNX, in percentage points
#   OIL    log return of Brent front-month BZ=F (WTI went negative in 2020)
#   COPPER log return of HG=F
#   GOLD   log return of GC=F
#   CREDIT log return HYG minus log return IEF (credit excess return)
#   EM     log return EEM minus log return ACWI
FACTOR_PROXIES: dict[str, list[str]] = {
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
# Factors whose unit is "percentage points of yield" instead of log return.
LEVEL_FACTORS = {"RATES"}

MIN_HISTORY_WEEKS = 104        # assets with less overlapping history are not modelled
ESTIMATION_WINDOW_WEEKS = 52 * 15
N_STAT_FACTORS = 4             # PCA factors extracted from macro-model residuals
PCA_MIN_COVERAGE = 0.85        # fraction of the window an asset needs to enter the PCA

# ---------------------------------------------------------------- regimes
N_REGIMES = 3
HMM_FACTORS = ["MKT", "CREDIT", "RATES", "USD"]
HMM_RESTARTS = 6

# ---------------------------------------------------------------- simulation
# Forward-looking factor premia: annual, ARITHMETIC, in excess of cash.
# Only factors with a defensible long-run premium get one. These drive every
# expected return in the engine — historical alphas are *not* extrapolated.
FACTOR_PREMIA_ANNUAL: dict[str, float] = {
    "MKT": 0.045,     # global equity risk premium
    "CREDIT": 0.010,  # HY minus Treasury, after defaults
}
# Asset classes that earn the cash rate on top of their factor premia.
CASH_CARRY_CLASSES = {"equity_index", "equity_country", "equity_sector", "equity_stock", "bond", "crypto"}
BLOCK_LENGTH_WEEKS = 8         # mean block length of the stationary bootstrap
N_PATHS = 4000
SEED = 7

FAN_TICKERS = ["ACWI", "SPY", "QQQ", "VGK", "EEM", "IEF", "GC=F", "BZ=F", "BTC-USD"]

# ---------------------------------------------------------------- rates
DNS_LAMBDA_MONTHLY = 0.0609    # Diebold-Li (2006), maturities in MONTHS
RATES_HORIZONS_MONTHS = {"3M": 3, "6M": 6, "1Y": 12, "2Y": 24, "3Y": 36, "5Y": 60}

# ---------------------------------------------------------------- backtest
BACKTEST_HORIZON_WEEKS = 52
BACKTEST_STEP_WEEKS = 26
BACKTEST_START = "2016-01-01"
BACKTEST_N_PATHS = 1500
BACKTEST_MAX_ASSETS = 80

# ---------------------------------------------------------------- index rules
# S&P 500 eligibility (S&P U.S. Indices Methodology). VERIFY the current threshold.
SP500_MIN_MCAP_USD = 22.7e9
SP500_IPO_SEASONING_MONTHS = 12
# Nasdaq-100 annual reconstitution: ranking on last trading day of October;
# members ranked 101–125 are retained if they were members. VERIFY the methodology.
NDX_SIZE = 100
NDX_BUFFER_RANK = 125
NDX_SEASONING_MONTHS = 3

# ---------------------------------------------------------------- rebalancing
REBAL_ABS_BAND = 0.05   # Swensen 5/25: absolute band of 5 percentage points…
REBAL_REL_BAND = 0.25   # …or 25% of the target weight, whichever is tighter


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
