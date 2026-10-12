"""
quant.data.loader — Robust market data loader with disk caching and frequency conversions.
Handles yfinance batch downloading, timezone stripping, missing data, and weekly returns.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

from quant.config import PRICE_START, CACHE_DIR, CACHE_MAX_AGE_HOURS

logger = logging.getLogger(__name__)
PRICES_CACHE_DIR = CACHE_DIR / "prices"
PRICES_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("^", "IDX_").replace("=", "EQ_").replace("/", "_")
    return PRICES_CACHE_DIR / f"{safe}.pkl"


def load_prices(
    tickers: List[str],
    start: str = PRICE_START,
    refresh: bool = False
) -> pd.DataFrame:
    """
    Loads daily adjusted close prices for tickers.
    Uses disk caching per ticker to enable fast incremental loading and high reliability.
    """
    tickers = sorted(list(set(tickers)))
    now = time.time()
    max_age_sec = CACHE_MAX_AGE_HOURS * 3600

    loaded: dict[str, pd.Series] = {}
    missing_tickers: List[str] = []

    for t in tickers:
        cp = _cache_path(t)
        if not refresh and cp.exists() and (now - cp.stat().st_mtime < max_age_sec):
            try:
                s = pd.read_pickle(cp)
                if isinstance(s, pd.Series) and not s.empty:
                    loaded[t] = s
                    continue
            except Exception:
                pass
        missing_tickers.append(t)

    # Batch download missing tickers in chunks
    if missing_tickers:
        chunk_size = 60
        for i in range(0, len(missing_tickers), chunk_size):
            chunk = missing_tickers[i:i + chunk_size]
            try:
                data = yf.download(
                    chunk,
                    start=start,
                    auto_adjust=True,
                    progress=False,
                    threads=True
                )
                if data is not None and not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        # MultiIndex with Close / Volume / etc.
                        if "Close" in data.columns.levels[0]:
                            close_df = data["Close"]
                        else:
                            close_df = data.xs("Close", axis=1, level=0, drop_level=True)
                    else:
                        # Single ticker dataframe
                        close_df = data[["Close"]] if "Close" in data.columns else data
                        if len(chunk) == 1:
                            close_df.columns = [chunk[0]]

                    # Strip timezone from index
                    if close_df.index.tz is not None:
                        close_df.index = close_df.index.tz_localize(None)

                    for t in chunk:
                        if t in close_df.columns:
                            col = close_df[t].dropna()
                            # Filter non-positive values
                            col = col[col > 0]
                            if not col.empty:
                                pd.to_pickle(col, _cache_path(t))
                                loaded[t] = col
            except Exception as e:
                logger.warning(f"Error downloading chunk {chunk}: {e}")

    if not loaded:
        return pd.DataFrame()

    combined = pd.DataFrame(loaded)
    combined.sort_index(inplace=True)
    if combined.index.tz is not None:
        combined.index = combined.index.tz_localize(None)

    # Filter out non-positive prices
    combined[combined <= 0] = np.nan
    return combined


def to_weekly(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples daily prices to weekly frequency (Friday close: W-FRI).
    Takes last valid observation per week.
    """
    if prices.empty:
        return pd.DataFrame()
    weekly = prices.resample("W-FRI").last()
    return weekly.dropna(how="all")


def weekly_log_returns(
    prices: pd.DataFrame,
    yield_tickers: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculates weekly returns:
    - Log returns for price series: log(P_t / P_{t-1})
    - Differences in percentage points for yield series (e.g. ^TNX)
    """
    if prices.empty:
        return pd.DataFrame()

    yield_set = set(yield_tickers or [])
    weekly = to_weekly(prices)
    returns = pd.DataFrame(index=weekly.index[1:], columns=weekly.columns, dtype=float)

    for col in weekly.columns:
        s = weekly[col]
        if col in yield_set:
            # Yield difference (e.g. ^TNX from 4.20 to 4.30 = +0.10)
            returns[col] = s.diff().iloc[1:]
        else:
            # Standard asset log return
            valid = (s > 0)
            log_s = np.log(s.where(valid))
            returns[col] = log_s.diff().iloc[1:]

    return returns.dropna(how="all")
