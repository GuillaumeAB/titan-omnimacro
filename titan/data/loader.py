"""
titan.data.loader — Market data ingestion and frequency resampling.
Uses disk caching with TTL to minimize redundant network requests.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from titan.config import CACHE_DIR, CACHE_MAX_AGE_HOURS, PRICE_START, WEEK_RULE

logger = logging.getLogger(__name__)
PRICES_CACHE_DIR = CACHE_DIR / "prices"
PRICES_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("^", "IDX_").replace("=", "EQ_").replace("/", "_")
    return PRICES_CACHE_DIR / f"{safe}.pkl"


def load_prices(
    tickers: Sequence[str],
    start: str = PRICE_START,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Loads daily adjusted close prices for tickers.
    Cached individually per ticker. Missing or failed tickers are omitted.
    """
    tickers = sorted(set(tickers))
    now = time.time()
    max_age_sec = CACHE_MAX_AGE_HOURS * 3600

    loaded: dict[str, pd.Series] = {}
    missing: list[str] = []

    for t in tickers:
        cp = _cache_path(t)
        if not refresh and cp.exists() and (now - cp.stat().st_mtime < max_age_sec):
            try:
                s = pd.read_pickle(cp)
                if isinstance(s, pd.Series) and not s.empty:
                    loaded[t] = s
                    continue
            except Exception as e:
                logger.debug("Failed to read cache for %s: %s", t, e)
        missing.append(t)

    if missing:
        chunk_size = 50
        for i in range(0, len(missing), chunk_size):
            chunk = missing[i : i + chunk_size]
            try:
                data = yf.download(
                    chunk,
                    start=start,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                if data is not None and not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        close_df = (
                            data["Close"]
                            if "Close" in data.columns.levels[0]
                            else data.xs("Close", axis=1, level=0, drop_level=True)
                        )
                    else:
                        close_df = data[["Close"]] if "Close" in data.columns else data
                        if len(chunk) == 1:
                            close_df.columns = [chunk[0]]

                    if close_df.index.tz is not None:
                        close_df.index = close_df.index.tz_localize(None)

                    for t in chunk:
                        if t in close_df.columns:
                            col = close_df[t].dropna()
                            col = col[col > 0]
                            if not col.empty:
                                pd.to_pickle(col, _cache_path(t))
                                loaded[t] = col
            except Exception as e:
                logger.warning("Error downloading tickers %s: %s", chunk, e)

    if not loaded:
        return pd.DataFrame()

    combined = pd.DataFrame(loaded).sort_index()
    if combined.index.tz is not None:
        combined.index = combined.index.tz_localize(None)

    combined[combined <= 0] = np.nan
    return combined


def to_weekly(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples daily prices to weekly frequency (Friday close: W-FRI).
    """
    if prices.empty:
        return pd.DataFrame()
    weekly = prices.resample(WEEK_RULE).last()
    return weekly.dropna(how="all")


def weekly_log_returns(
    prices: pd.DataFrame,
    yield_tickers: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Calculates weekly returns:
    - Log returns for price series: log(P_t / P_{t-1})
    - Level differences in pp for yield series (e.g., ^TNX).
    """
    if prices.empty:
        return pd.DataFrame()

    yield_set = set(yield_tickers)
    weekly = to_weekly(prices)
    returns = pd.DataFrame(index=weekly.index[1:], columns=weekly.columns, dtype=float)

    for col in weekly.columns:
        s = weekly[col]
        if col in yield_set:
            returns[col] = s.diff().iloc[1:]
        else:
            valid = s > 0
            log_s = np.log(s.where(valid))
            returns[col] = log_s.diff().iloc[1:]

    return returns.dropna(how="all")
