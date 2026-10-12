"""
quant.index.recon — Index Reconstitution & IPO Pipeline Predictor.
Implements institutional models for Nasdaq-100 December annual rebalance
(with official 100/125 buffer rule and Monte Carlo ranking), S&P 500 addition
eligibility (GAAP profitability, float, seasoning, migration from S&P 400),
and IPO index inclusion timeline mapping.
"""

from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

from quant.config import (
    SP500_MIN_MCAP_USD,
    SP500_MIN_FLOAT,
    SP500_IPO_SEASONING_MONTHS,
    NDX_SIZE,
    NDX_BUFFER_RANK,
    NDX_SEASONING_MONTHS
)


def predict_nasdaq100_reconstitution(
    screener: pd.DataFrame,
    members: pd.DataFrame,
    prices: Optional[pd.DataFrame] = None,
    n_sims: int = 5000,
    days_to_reference: Optional[int] = None
) -> pd.DataFrame:
    """
    Predicts Nasdaq-100 December reconstitution outcomes.
    Rules:
    - Non-Financials only (screener sector != 'Finance')
    - Top 100 ranked by market cap are included
    - Current members ranked 101-125 are retained (buffer rule)
    - Current members ranked > 125 are deleted
    - Non-members ranked <= 100 replace departing members
    Simulates forward to reference date (late October) via lognormal Monte Carlo.
    """
    if screener.empty or members.empty:
        return pd.DataFrame()

    member_set = set(members["ticker"].str.upper())

    # Filter screener: Nasdaq exchange, non-finance, valid market cap
    df = screener.copy()
    df["ticker"] = df["ticker"].str.upper()

    # Sector filter: exclude Financials
    non_fin = df[~df["sector"].astype(str).str.lower().str.contains("finance|financial|bank|insurance", na=False)]
    non_fin = non_fin[non_fin["market_cap"].notna() & (non_fin["market_cap"] > 1e8)].copy()

    # Deduplicate share classes by taking largest market cap per root symbol
    non_fin["root"] = non_fin["ticker"].apply(lambda s: s.split("-")[0].split(".")[0])
    non_fin = non_fin.sort_values("market_cap", ascending=False).drop_duplicates(subset=["root"]).copy()

    N = len(non_fin)
    if N == 0:
        return pd.DataFrame()

    non_fin["is_member"] = non_fin["ticker"].isin(member_set)
    non_fin["rank_now"] = np.arange(1, N + 1)

    # Days to next reference date (last trading day of October)
    if days_to_reference is None:
        today = datetime.now()
        ref_year = today.year if today.month <= 10 else today.year + 1
        ref_date = datetime(ref_year, 10, 31)
        days_to_reference = max(5, (ref_date - today).days)

    t_years = days_to_reference / 365.0

    # Stock-specific annual volatility from prices if provided, else default 40%
    mcaps = non_fin["market_cap"].values  # (N,)
    vols = np.full(N, 0.40)
    if prices is not None and not prices.empty:
        for i, t in enumerate(non_fin["ticker"]):
            if t in prices.columns:
                s = prices[t].dropna()
                if len(s) > 20:
                    ret = np.log(s).diff().dropna()
                    vols[i] = float(np.std(ret) * np.sqrt(252.0))

    # Monte Carlo lognormal simulation of market caps to reference date
    rng = np.random.default_rng(42)
    # Market factor shock m ~ N(0, (0.18)^2 * t)
    m_shock = rng.normal(0, 0.18 * np.sqrt(t_years), size=n_sims)  # (n_sims,)
    idiosyncratic = rng.normal(0, vols * np.sqrt(t_years), size=(n_sims, N))  # (n_sims, N)

    # Simulated market caps: mcap_0 * exp(m + eps - 0.5*sigma^2*t)
    drift = -0.5 * (vols ** 2) * t_years
    log_changes = m_shock[:, None] + idiosyncratic + drift[None, :]
    sim_mcaps = mcaps[None, :] * np.exp(log_changes)  # (n_sims, N)

    # Rank per simulation (descending, 0-indexed)
    # argsort of -sim_mcaps gives indices of largest to smallest
    sim_ranks = np.zeros((n_sims, N), dtype=int)
    for s in range(n_sims):
        order = np.argsort(-sim_mcaps[s])
        sim_ranks[s, order] = np.arange(1, N + 1)

    # Apply official Nasdaq-100 reconstitution buffer rule:
    # A stock is included if:
    #   rank <= 100
    #   OR (is_member AND rank <= 125)
    is_mem_arr = non_fin["is_member"].values  # (N,)
    included = (sim_ranks <= NDX_SIZE) | (is_mem_arr[None, :] & (sim_ranks <= NDX_BUFFER_RANK))
    prob_in_index = np.mean(included, axis=0)  # (N,)

    non_fin["prob_in_index"] = [round(float(p), 4) for p in prob_in_index]

    # Assign predicted action
    actions = []
    for _, r in non_fin.iterrows():
        p = r["prob_in_index"]
        is_mem = r["is_member"]
        rank = r["rank_now"]

        if is_mem:
            if p >= 0.85:
                actions.append("KEEP")
            elif p <= 0.25:
                actions.append("DELETE")
            else:
                actions.append("WATCH-DELETE")
        else:
            if p >= 0.70:
                actions.append("ADD")
            elif p >= 0.30:
                actions.append("WATCH-ADD")
            else:
                actions.append("NON-MEMBER")

    non_fin["predicted_action"] = actions
    keep_cols = ["ticker", "name", "market_cap", "rank_now", "is_member", "prob_in_index", "predicted_action"]
    return non_fin[keep_cols].sort_values("rank_now").reset_index(drop=True)


def sp500_addition_candidates(
    screeners: pd.DataFrame,
    sp500: pd.DataFrame,
    sp400: pd.DataFrame,
    prices: Optional[pd.DataFrame] = None,
    check_profitability: bool = True,
    top_n: int = 40
) -> pd.DataFrame:
    """
    Identifies and scores non-member companies eligible for addition to S&P 500.
    Criteria:
    - US domicile & US listed
    - Market cap >= threshold ($22.7B)
    - Positive GAAP earnings (sum of trailing 4 quarters > 0 and latest quarter > 0)
    - IPO seasoning >= 12 months
    - Often migrated from S&P 400 MidCap
    """
    if screeners.empty or sp500.empty:
        return pd.DataFrame()

    sp500_set = set(sp500["ticker"].str.upper())
    sp400_set = set(sp400["ticker"].str.upper()) if not sp400.empty else set()

    # Filter for US non-members
    df = screeners.copy()
    df["ticker"] = df["ticker"].str.upper()
    non_members = df[~df["ticker"].isin(sp500_set)].copy()

    # Filter US country
    us_mask = non_members["country"].astype(str).str.lower().isin(["united states", "us", "usa"])
    non_members = non_members[us_mask & (non_members["market_cap"].notna())].copy()

    # Market cap threshold (~70% of current $22.7B threshold = ~$16B)
    min_cap = SP500_MIN_MCAP_USD * 0.70
    candidates = non_members[non_members["market_cap"] >= min_cap].sort_values("market_cap", ascending=False).head(top_n).copy()

    candidates["in_sp400"] = candidates["ticker"].isin(sp400_set)

    # IPO seasoning check
    current_year = datetime.now().year
    candidates["seasoning_ok"] = candidates["ipo_year"].isna() | (candidates["ipo_year"] <= current_year - 1)

    # Profitability check via yfinance for top candidates
    profitable = []
    for t in candidates["ticker"]:
        is_prof = True
        if check_profitability:
            try:
                tk = yf.Ticker(t)
                stmt = tk.quarterly_income_stmt
                if stmt is not None and not stmt.empty:
                    # Look for Net Income row
                    net_inc_rows = [r for r in stmt.index if "net income" in str(r).lower()]
                    if net_inc_rows:
                        vals = stmt.loc[net_inc_rows[0]].dropna().values[:4]
                        if len(vals) >= 2:
                            is_prof = bool(np.sum(vals) > 0 and vals[0] > 0)
            except Exception:
                pass
        profitable.append(is_prof)

    candidates["profitable"] = profitable

    # Score calculation based on mcap ratio, S&P 400 membership, and profitability
    mcap_ratio = candidates["market_cap"] / SP500_MIN_MCAP_USD
    candidates["score"] = (
        mcap_ratio * 40.0 +
        candidates["in_sp400"].astype(int) * 30.0 +
        candidates["profitable"].astype(int) * 20.0 +
        candidates["seasoning_ok"].astype(int) * 10.0
    ).round(1)

    candidates["prob_above_threshold_6m"] = np.clip(mcap_ratio * 0.8, 0.05, 0.99).round(2)

    cols = ["ticker", "name", "market_cap", "exchange", "in_sp400", "profitable", "seasoning_ok", "prob_above_threshold_6m", "score"]
    return candidates[[c for c in cols if c in candidates.columns]].sort_values("score", ascending=False).reset_index(drop=True)


def sp500_deletion_risk(
    screeners: pd.DataFrame,
    sp500: pd.DataFrame,
    top_n: int = 30
) -> pd.DataFrame:
    """
    Identifies smallest S&P 500 member companies at risk of deletion
    due to market capitalization falling far below entry criteria.
    """
    if screeners.empty or sp500.empty:
        return pd.DataFrame()

    sp500_set = set(sp500["ticker"].str.upper())
    df = screeners.copy()
    df["ticker"] = df["ticker"].str.upper()

    members = df[df["ticker"].isin(sp500_set) & df["market_cap"].notna()].copy()
    members = members.sort_values("market_cap", ascending=True).head(top_n).copy()

    members["ratio_to_threshold"] = (members["market_cap"] / SP500_MIN_MCAP_USD).round(3)
    members["deletion_risk"] = np.where(
        members["market_cap"] < (SP500_MIN_MCAP_USD * 0.35), "HIGH",
        np.where(members["market_cap"] < (SP500_MIN_MCAP_USD * 0.55), "MODERATE", "LOW")
    )

    cols = ["ticker", "name", "market_cap", "sector", "ratio_to_threshold", "deletion_risk"]
    return members[[c for c in cols if c in members.columns]].reset_index(drop=True)


def ipo_index_timeline(ipo_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Maps recently priced and filed IPOs to their earliest index eligibility dates:
    - S&P 500: Earliest = Pricing Date + 12 months (1-year seasoning rule)
    - Nasdaq-100: Earliest = Pricing Date + 3 months
    """
    priced = ipo_frames.get("priced", pd.DataFrame())
    if priced.empty:
        return pd.DataFrame()

    rows = []
    for _, r in priced.iterrows():
        ticker = r.get("ticker", "")
        name = r.get("name", "")
        exchange = r.get("exchange", "")
        offer = r.get("offer_amount", 0.0)
        date_str = r.get("date", "")

        sp500_eligible = "TBD"
        ndx_eligible = "TBD"

        try:
            d = pd.to_datetime(date_str)
            sp500_eligible = (d + pd.DateOffset(months=SP500_IPO_SEASONING_MONTHS)).strftime("%Y-%m")
            ndx_eligible = (d + pd.DateOffset(months=NDX_SEASONING_MONTHS)).strftime("%Y-%m")
        except Exception:
            pass

        rows.append({
            "ticker": ticker,
            "company": name,
            "exchange": exchange,
            "pricing_date": date_str,
            "offer_amount_usd": offer,
            "sp500_earliest_date": sp500_eligible,
            "ndx_earliest_date": ndx_eligible,
            "notes": "Large Offering" if offer and offer > 5e8 else "Standard IPO"
        })

    return pd.DataFrame(rows)
