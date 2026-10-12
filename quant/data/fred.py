"""
quant.data.fred — Macroeconomic & Treasury yield curve data from St. Louis FRED.
Direct CSV downloads with zero API key dependency and local disk caching.
"""

import io
import time
import logging
from pathlib import Path
from typing import List
import pandas as pd
import requests

from quant.config import CACHE_DIR, CACHE_MAX_AGE_HOURS, HTTP_HEADERS

logger = logging.getLogger(__name__)
FRED_CACHE_DIR = CACHE_DIR / "fred"
FRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

TREASURY_SERIES = {
    0.25: "DGS3MO",
    1.0: "DGS1",
    2.0: "DGS2",
    3.0: "DGS3",
    5.0: "DGS5",
    7.0: "DGS7",
    10.0: "DGS10",
    20.0: "DGS20",
    30.0: "DGS30"
}


def load_fred_series(
    series_ids: List[str],
    start: str = "1990-01-01"
) -> pd.DataFrame:
    """
    Downloads and caches FRED series via direct fredgraph.csv URL.
    Returns DataFrame indexed by DatetimeIndex with float columns.
    """
    cache_file = FRED_CACHE_DIR / f"{'_'.join(sorted(series_ids))}.pkl"
    now = time.time()
    max_age_sec = CACHE_MAX_AGE_HOURS * 3600

    if cache_file.exists() and (now - cache_file.stat().st_mtime < max_age_sec):
        try:
            df = pd.read_pickle(cache_file)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            pass

    series_query = ",".join(series_ids)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_query}"

    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=20)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True, na_values=".")
            df.index = pd.to_datetime(df.index)
            # Ensure numeric
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[df.index >= pd.to_datetime(start)]
            pd.to_pickle(df, cache_file)
            return df
        else:
            logger.warning(f"FRED HTTP {resp.status_code} for series {series_ids}")
    except Exception as e:
        logger.warning(f"Failed to fetch FRED series {series_ids}: {e}")

    # Fallback to cache if available even if older
    if cache_file.exists():
        return pd.read_pickle(cache_file)
    return pd.DataFrame()


def load_treasury_curve(start: str = "1990-01-01") -> pd.DataFrame:
    """
    Returns US Treasury yield curve as DataFrame with float maturities as columns:
    [0.25, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0] in percent (e.g. 4.25 = 4.25%).
    Falls back to Yahoo Finance Treasury yield indices if FRED is unreachable.
    """
    series_ids = list(TREASURY_SERIES.values())
    raw = load_fred_series(series_ids, start=start)
    if not raw.empty:
        id_to_mat = {v: k for k, v in TREASURY_SERIES.items()}
        renamed = {}
        for col in raw.columns:
            if col in id_to_mat:
                renamed[col] = id_to_mat[col]

        curve = raw[list(renamed.keys())].rename(columns=renamed)
        curve = curve.reindex(columns=sorted(list(TREASURY_SERIES.keys())))
        curve = curve.dropna(how="all")
        if not curve.empty:
            return curve

    # Fallback to Yahoo Finance Treasury Yield indices
    try:
        from quant.data.loader import load_prices
        y_tickers = ["^IRX", "^FVX", "^TNX", "^TYX"]
        yp = load_prices(y_tickers, start=start)
        if not yp.empty:
            # Map ^IRX -> 0.25y, ^FVX -> 5y, ^TNX -> 10y, ^TYX -> 30y
            curve = pd.DataFrame(index=yp.index)
            if "^IRX" in yp.columns: curve[0.25] = yp["^IRX"]
            if "^FVX" in yp.columns: curve[5.0] = yp["^FVX"]
            if "^TNX" in yp.columns: curve[10.0] = yp["^TNX"]
            if "^TYX" in yp.columns: curve[30.0] = yp["^TYX"]

            # Interpolate missing maturities linearly across columns
            all_mats = sorted(list(TREASURY_SERIES.keys()))
            curve = curve.reindex(columns=all_mats).interpolate(axis=1, method="linear").bfill(axis=1).ffill(axis=1)
            return curve.dropna(how="all")
    except Exception as e:
        logger.warning(f"Yahoo yields fallback error: {e}")

    return pd.DataFrame()


def latest_risk_free() -> float:
    """
    Returns current annual risk-free rate as a decimal (e.g. 0.051 = 5.1%).
    Primary source: FRED DGS3MO (3-Month T-Bill).
    Fallback: 0.045
    """
    try:
        df = load_fred_series(["DGS3MO"])
        if not df.empty and "DGS3MO" in df.columns:
            val = df["DGS3MO"].dropna().iloc[-1]
            if 0.0 < val < 25.0:
                return float(val / 100.0)
    except Exception:
        pass
    return 0.045
