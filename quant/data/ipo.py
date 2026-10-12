"""
quant.data.ipo — IPO calendar intelligence from Nasdaq official API.
Retrieves and normalizes priced, upcoming, and filed IPO deals.
"""

import re
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import requests

from quant.config import CACHE_DIR, CACHE_MAX_AGE_HOURS, HTTP_HEADERS

logger = logging.getLogger(__name__)
IPO_CACHE_DIR = CACHE_DIR / "ipo"
IPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _parse_price_range(val: str) -> Optional[float]:
    if not val or pd.isna(val):
        return None
    s = str(val).replace("$", "").replace(",", "").strip()
    if "-" in s:
        parts = s.split("-")
        try:
            p1, p2 = float(parts[0].strip()), float(parts[1].strip())
            return (p1 + p2) / 2.0
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_numeric(val: str) -> Optional[float]:
    if not val or pd.isna(val):
        return None
    clean = re.sub(r"[^\d.]", "", str(val))
    try:
        return float(clean)
    except Exception:
        return None


def get_ipo_calendar(months: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Retrieves IPO pipeline across priced, upcoming, and filed deals.
    Returns dict with keys: 'priced', 'upcoming', 'filed'.
    """
    if months is None:
        now = datetime.now()
        # Previous, current, and next month
        prev_m = 12 if now.month == 1 else now.month - 1
        prev_y = now.year - 1 if now.month == 1 else now.year
        next_m = 1 if now.month == 12 else now.month + 1
        next_y = now.year + 1 if now.month == 12 else now.year

        months = [
            f"{prev_y}-{prev_m:02d}",
            f"{now.year}-{now.month:02d}",
            f"{next_y}-{next_m:02d}"
        ]

    cache_file = IPO_CACHE_DIR / f"ipo_{'_'.join(months)}.pkl"
    now_ts = time.time()
    if cache_file.exists() and (now_ts - cache_file.stat().st_mtime < CACHE_MAX_AGE_HOURS * 3600):
        try:
            return pd.read_pickle(cache_file)
        except Exception:
            pass

    combined: Dict[str, List[dict]] = {"priced": [], "upcoming": [], "filed": []}

    for m in months:
        url = f"https://api.nasdaq.com/api/ipo/calendar?date={m}"
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=15)
            if resp.status_code == 200:
                js = resp.json()
                data = js.get("data", {})
                for cat in ["priced", "upcoming", "filed"]:
                    rows = data.get(cat, {}).get("rows", [])
                    if rows:
                        for r in rows:
                            ticker = r.get("proposedTickerSymbol") or r.get("symbol") or ""
                            cname = r.get("companyName") or ""
                            exchange = r.get("proposedExchange") or r.get("exchange") or ""
                            price_raw = r.get("proposedSharePrice") or r.get("price") or ""
                            shares_raw = r.get("sharesOffered") or ""
                            amount_raw = r.get("dollarValueOfSharesOffered") or ""
                            date_str = r.get("pricedDate") or r.get("expectedDate") or r.get("filedDate") or ""

                            combined[cat].append({
                                "ticker": str(ticker).strip(),
                                "name": str(cname).strip(),
                                "exchange": str(exchange).strip(),
                                "price": _parse_price_range(price_raw),
                                "shares": _parse_numeric(shares_raw),
                                "offer_amount": _parse_numeric(amount_raw),
                                "date": str(date_str).strip(),
                                "month": m
                            })
        except Exception as e:
            logger.warning(f"Error fetching IPO calendar for {m}: {e}")

    result = {}
    for cat in ["priced", "upcoming", "filed"]:
        if combined[cat]:
            df = pd.DataFrame(combined[cat]).drop_duplicates(subset=["ticker", "name"]).reset_index(drop=True)
        else:
            df = pd.DataFrame(columns=["ticker", "name", "exchange", "price", "shares", "offer_amount", "date", "month"])
        result[cat] = df

    pd.to_pickle(result, cache_file)
    return result
