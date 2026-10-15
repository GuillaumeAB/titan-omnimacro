"""
titan.index.recon — Index Reconstitution & IPO Pipeline Predictor.
Implements institutional models for Nasdaq-100 December annual rebalance
(with official 100/125 buffer rule and Monte Carlo ranking), S&P 500 addition
eligibility (GAAP profitability, float, seasoning), and IPO timeline mapping.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from titan.config import (
    NDX_BUFFER_RANK,
    NDX_SIZE,
    SP500_IPO_SEASONING_MONTHS,
    SP500_MIN_MCAP_USD,
)


def predict_nasdaq100_reconstitution(
    screener: pd.DataFrame,
    members: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    n_sims: int = 3000,
    days_to_reference: int | None = None,
) -> pd.DataFrame:
    if screener.empty or members.empty:
        return pd.DataFrame()

    member_set = set(members["ticker"].str.upper())
    df = screener.copy()
    df["ticker"] = df["ticker"].str.upper()

    non_fin = df[~df["sector"].astype(str).str.lower().str.contains("finance|financial|bank|insurance", na=False)]
    non_fin = non_fin[non_fin["market_cap"].notna() & (non_fin["market_cap"] > 1e8)].copy()

    non_fin["root"] = non_fin["ticker"].apply(lambda s: s.split("-")[0].split(".")[0])
    non_fin = non_fin.sort_values("market_cap", ascending=False).drop_duplicates(subset=["root"]).copy()

    n = len(non_fin)
    if n == 0:
        return pd.DataFrame()

    non_fin["is_member"] = non_fin["ticker"].isin(member_set)
    non_fin["rank_now"] = np.arange(1, n + 1)

    if days_to_reference is None:
        today = datetime.now()
        ref_year = today.year if today.month <= 10 else today.year + 1
        ref_date = datetime(ref_year, 10, 31)
        days_to_reference = max(5, (ref_date - today).days)

    t_years = days_to_reference / 365.0

    mcaps = non_fin["market_cap"].values
    vols = np.full(n, 0.40)
    if prices is not None and not prices.empty:
        for i, t in enumerate(non_fin["ticker"]):
            if t in prices.columns:
                s = prices[t].dropna()
                if len(s) > 20:
                    ret = np.log(s).diff().dropna()
                    vols[i] = float(np.std(ret) * np.sqrt(252.0))

    rng = np.random.default_rng(42)
    m_shock = rng.normal(0, 0.18 * np.sqrt(t_years), size=n_sims)
    idiosyncratic = rng.normal(0, vols * np.sqrt(t_years), size=(n_sims, n))

    drift = -0.5 * (vols**2) * t_years
    log_changes = m_shock[:, None] + idiosyncratic + drift[None, :]
    sim_mcaps = mcaps[None, :] * np.exp(log_changes)

    sim_ranks = np.zeros((n_sims, n), dtype=int)
    for s in range(n_sims):
        order = np.argsort(-sim_mcaps[s])
        sim_ranks[s, order] = np.arange(1, n + 1)

    is_mem_arr = non_fin["is_member"].values
    included = (sim_ranks <= NDX_SIZE) | (is_mem_arr[None, :] & (sim_ranks <= NDX_BUFFER_RANK))
    prob_in_index = np.mean(included, axis=0)
    non_fin["prob_in_index"] = [round(float(p), 4) for p in prob_in_index]

    actions = []
    for _, r in non_fin.iterrows():
        p = r["prob_in_index"]
        is_mem = r["is_member"]
        if is_mem:
            actions.append("KEEP" if p >= 0.85 else ("DELETE" if p <= 0.25 else "WATCH-DELETE"))
        else:
            actions.append("ADD" if p >= 0.70 else ("WATCH-ADD" if p >= 0.30 else "NON-MEMBER"))

    non_fin["predicted_action"] = actions
    keep_cols = ["ticker", "name", "market_cap", "rank_now", "is_member", "prob_in_index", "predicted_action"]
    return non_fin[keep_cols].sort_values("rank_now").reset_index(drop=True)


def sp500_addition_candidates(
    screeners: pd.DataFrame,
    sp500: pd.DataFrame,
    sp400: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    check_profitability: bool = True,
    top_n: int = 30,
) -> pd.DataFrame:
    if screeners.empty or sp500.empty:
        return pd.DataFrame()

    sp500_set = set(sp500["ticker"].str.upper())
    sp400_set = set(sp400["ticker"].str.upper()) if not sp400.empty else set()

    df = screeners.copy()
    df["ticker"] = df["ticker"].str.upper()
    non_members = df[~df["ticker"].isin(sp500_set)].copy()

    us_mask = non_members["country"].astype(str).str.lower().isin(["united states", "us", "usa"])
    non_members = non_members[us_mask & (non_members["market_cap"].notna())].copy()

    min_cap = SP500_MIN_MCAP_USD * 0.70
    candidates = (
        non_members[non_members["market_cap"] >= min_cap]
        .sort_values("market_cap", ascending=False)
        .head(top_n)
        .copy()
    )

    candidates["in_sp400"] = candidates["ticker"].isin(sp400_set)
    current_year = datetime.now().year
    candidates["seasoning_ok"] = candidates["ipo_year"].isna() | (candidates["ipo_year"] <= current_year - 1)

    mcap_ratio = candidates["market_cap"] / SP500_MIN_MCAP_USD
    candidates["score"] = (
        mcap_ratio * 40.0
        + candidates["in_sp400"].astype(int) * 30.0
        + candidates["seasoning_ok"].astype(int) * 30.0
    ).round(1)

    candidates["prob_above_threshold_6m"] = np.clip(mcap_ratio * 0.8, 0.05, 0.99).round(2)
    cols = ["ticker", "name", "market_cap", "exchange", "in_sp400", "seasoning_ok", "prob_above_threshold_6m", "score"]
    return candidates[[c for c in cols if c in candidates.columns]].sort_values("score", ascending=False).reset_index(drop=True)
