"""
titan.data.polymarket — Real Polymarket Gamma API integration.
Fetches active prediction market odds, volume, and implied probabilities
with UTC timestamps and transparent error/connectivity reporting.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from titan.config import CACHE_DIR, CACHE_MAX_AGE_HOURS, HTTP_HEADERS, HTTP_TIMEOUT

logger = logging.getLogger(__name__)
POLY_CACHE_FILE = CACHE_DIR / "polymarket_events.json"

GAMMA_API_URL = "https://gamma-api.polymarket.com/events"

# Curated relevant macro/geopolitical query keywords
MACRO_KEYWORDS = ["Taiwan", "Fed", "Interest Rates", "Oil", "Ukraine", "Recession", "Tariff", "AI"]


@dataclass
class MarketOdds:
    market_id: str
    question: str
    slug: str
    implied_probability: float
    volume_usd: float
    end_date: str
    source: str
    as_of: str
    is_live: bool


def fetch_polymarket_events(limit: int = 15, refresh: bool = False) -> list[MarketOdds]:
    """
    Fetches real events from Polymarket Gamma API.
    Extracts implied yes/no probabilities, trading volumes, and timestamps.
    Returns list of MarketOdds with live flag.
    """
    now = time.time()
    max_age_sec = CACHE_MAX_AGE_HOURS * 3600

    if not refresh and POLY_CACHE_FILE.exists() and (now - POLY_CACHE_FILE.stat().st_mtime < max_age_sec):
        try:
            with open(POLY_CACHE_FILE, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                return [MarketOdds(**item) for item in cached_data]
        except Exception as e:
            logger.debug("Failed reading Polymarket cache: %s", e)

    results: list[MarketOdds] = []
    as_of_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        params = {
            "closed": "false",
            "limit": limit,
            "order": "volume24hr",
            "ascending": "false",
        }
        resp = requests.get(GAMMA_API_URL, params=params, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        if resp.status_code == 200:
            events = resp.json()
            for ev in events:
                markets = ev.get("markets", [])
                for m in markets:
                    q = m.get("question") or ev.get("title", "")
                    slug = m.get("slug") or ev.get("slug", "")
                    vol = float(m.get("volume") or ev.get("volume") or 0.0)
                    end_d = m.get("endDate") or ev.get("endDate") or ""

                    # Outcome prices parsing
                    prob = 0.5
                    outcome_prices = m.get("outcomePrices")
                    if outcome_prices:
                        if isinstance(outcome_prices, str):
                            try:
                                parsed = json.loads(outcome_prices)
                                prob = float(parsed[0])
                            except Exception:
                                pass
                        elif isinstance(outcome_prices, list) and len(outcome_prices) > 0:
                            try:
                                prob = float(outcome_prices[0])
                            except Exception:
                                pass

                    mid = str(m.get("id") or ev.get("id"))
                    results.append(
                        MarketOdds(
                            market_id=mid,
                            question=q,
                            slug=slug,
                            implied_probability=round(prob, 4),
                            volume_usd=round(vol, 2),
                            end_date=end_d,
                            source="Polymarket Gamma API",
                            as_of=as_of_str,
                            is_live=True,
                        )
                    )

            if results:
                with open(POLY_CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump([asdict(r) for r in results], f, indent=2)
                return results

    except Exception as e:
        logger.warning("Polymarket API unreachable: %s", e)

    # Fallback to cache if exists
    if POLY_CACHE_FILE.exists():
        try:
            with open(POLY_CACHE_FILE, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                out = []
                for item in cached_data:
                    item["is_live"] = False
                    out.append(MarketOdds(**item))
                return out
        except Exception:
            pass

    return []
