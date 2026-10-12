"""
quant.data.membership — Index membership constituents and exchange stock screeners.
Parses S&P 500, S&P 400, Nasdaq-100, and comprehensive Nasdaq/NYSE screeners with disk caching.
"""

import io
import time
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import requests

from quant.config import CACHE_DIR, CACHE_MAX_AGE_HOURS, HTTP_HEADERS

logger = logging.getLogger(__name__)
MEMBERSHIP_CACHE_DIR = CACHE_DIR / "membership"
MEMBERSHIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_wikipedia_tables(url: str, retries: int = 3, timeout: int = 25) -> list[pd.DataFrame]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
            if resp.status_code == 200:
                tables = pd.read_html(io.StringIO(resp.text))
                if tables:
                    return tables
        except Exception as e:
            time.sleep(1.5 * (attempt + 1))
    return []


def get_sp500() -> pd.DataFrame:
    """Returns S&P 500 constituents: ticker, name, sector, date_added."""
    cache_file = MEMBERSHIP_CACHE_DIR / "sp500.pkl"
    now = time.time()
    if cache_file.exists() and (now - cache_file.stat().st_mtime < CACHE_MAX_AGE_HOURS * 3600):
        try:
            return pd.read_pickle(cache_file)
        except Exception:
            pass

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _fetch_wikipedia_tables(url)
    if tables:
        t = tables[0]
        # Rename standard columns
        col_map = {}
        for c in t.columns:
            cl = str(c).lower()
            if "symbol" in cl or "ticker" in cl:
                col_map[c] = "ticker"
            elif "security" in cl or "name" in cl or "company" in cl:
                col_map[c] = "name"
            elif "gics sector" in cl or "sector" in cl:
                col_map[c] = "sector"
            elif "date added" in cl or "date" in cl:
                col_map[c] = "date_added"

        df = t.rename(columns=col_map)
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()
            df = df[["ticker", "name", "sector", "date_added"] if "date_added" in df.columns else ["ticker", "name", "sector"]]
            pd.to_pickle(df, cache_file)
            return df

    if cache_file.exists():
        return pd.read_pickle(cache_file)
    return pd.DataFrame(columns=["ticker", "name", "sector", "date_added"])


def get_sp400() -> pd.DataFrame:
    """Returns S&P 400 MidCap constituents: ticker, name, sector."""
    cache_file = MEMBERSHIP_CACHE_DIR / "sp400.pkl"
    now = time.time()
    if cache_file.exists() and (now - cache_file.stat().st_mtime < CACHE_MAX_AGE_HOURS * 3600):
        try:
            return pd.read_pickle(cache_file)
        except Exception:
            pass

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    tables = _fetch_wikipedia_tables(url)
    if tables:
        t = tables[0]
        col_map = {}
        for c in t.columns:
            cl = str(c).lower()
            if "symbol" in cl or "ticker" in cl:
                col_map[c] = "ticker"
            elif "security" in cl or "name" in cl or "company" in cl:
                col_map[c] = "name"
            elif "gics sector" in cl or "sector" in cl:
                col_map[c] = "sector"

        df = t.rename(columns=col_map)
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()
            cols = [c for c in ["ticker", "name", "sector"] if c in df.columns]
            df = df[cols]
            pd.to_pickle(df, cache_file)
            return df

    if cache_file.exists():
        return pd.read_pickle(cache_file)
    return pd.DataFrame(columns=["ticker", "name", "sector"])


def get_nasdaq100() -> pd.DataFrame:
    """Returns Nasdaq-100 constituents: ticker, name."""
    cache_file = MEMBERSHIP_CACHE_DIR / "nasdaq100.pkl"
    now = time.time()
    if cache_file.exists() and (now - cache_file.stat().st_mtime < CACHE_MAX_AGE_HOURS * 3600):
        try:
            return pd.read_pickle(cache_file)
        except Exception:
            pass

    # Try Nasdaq official quote API endpoint first
    try:
        url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=15)
        if resp.status_code == 200:
            js = resp.json()
            rows = js.get("data", {}).get("data", {}).get("rows", [])
            if rows:
                df = pd.DataFrame(rows)
                df = df.rename(columns={"symbol": "ticker", "companyName": "name"})
                df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()
                res = df[["ticker", "name"]].drop_duplicates(subset=["ticker"]).reset_index(drop=True)
                pd.to_pickle(res, cache_file)
                return res
    except Exception as e:
        logger.warning(f"Nasdaq API list-type/nasdaq100 error: {e}")

    # Fallback to Wikipedia
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = _fetch_wikipedia_tables(url)
    for t in tables:
        col_names = [str(c).lower() for c in t.columns]
        if any("ticker" in c or "symbol" in c for c in col_names) and len(t) >= 95:
            col_map = {}
            for c in t.columns:
                cl = str(c).lower()
                if "ticker" in cl or "symbol" in cl:
                    col_map[c] = "ticker"
                elif "company" in cl or "security" in cl:
                    col_map[c] = "name"
            df = t.rename(columns=col_map)
            df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()
            res = df[["ticker", "name"]].drop_duplicates(subset=["ticker"]).reset_index(drop=True)
            pd.to_pickle(res, cache_file)
            return res

    if cache_file.exists():
        return pd.read_pickle(cache_file)
    return pd.DataFrame(columns=["ticker", "name"])


def get_nasdaq_screener(exchange: str = "nasdaq") -> pd.DataFrame:
    """
    Downloads comprehensive exchange screener from api.nasdaq.com:
    ticker, name, market_cap (float USD), last_price, volume, country, ipo_year, sector, industry, exchange.
    """
    ex = exchange.lower()
    cache_file = MEMBERSHIP_CACHE_DIR / f"screener_{ex}.pkl"
    now = time.time()
    if cache_file.exists() and (now - cache_file.stat().st_mtime < CACHE_MAX_AGE_HOURS * 3600):
        try:
            return pd.read_pickle(cache_file)
        except Exception:
            pass

    url = f"https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange={ex}&download=true"
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=25)
        if resp.status_code == 200:
            js = resp.json()
            rows = js.get("data", {}).get("rows", [])
            if rows:
                df = pd.DataFrame(rows)
                df = df.rename(columns={
                    "symbol": "ticker",
                    "marketCap": "market_cap",
                    "lastsale": "last_price",
                    "ipoyear": "ipo_year"
                })
                df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()
                # Parse numeric market cap ($3,450,000,000 or raw number)
                df["market_cap"] = df["market_cap"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
                df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
                df["last_price"] = df["last_price"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
                df["last_price"] = pd.to_numeric(df["last_price"], errors="coerce")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                df["ipo_year"] = pd.to_numeric(df["ipo_year"], errors="coerce")
                df["exchange"] = ex.upper()

                keep = ["ticker", "name", "market_cap", "last_price", "volume", "country", "ipo_year", "sector", "industry", "exchange"]
                res = df[[c for c in keep if c in df.columns]].dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
                pd.to_pickle(res, cache_file)
                return res
    except Exception as e:
        logger.warning(f"Failed to fetch screener for {exchange}: {e}")

    if cache_file.exists():
        return pd.read_pickle(cache_file)
    return pd.DataFrame(columns=["ticker", "name", "market_cap", "last_price", "volume", "country", "ipo_year", "sector", "industry", "exchange"])
