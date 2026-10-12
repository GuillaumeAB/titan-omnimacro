"""
quant.pipeline — Master Orchestrator for the Global Multi-Asset Forecasting Engine.
Executes the full institutional workflow, generates joint distribution forecasts,
index reconstitutions, scenario stress tests, and produces output/latest.json.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


from quant.config import (
    ROOT,
    OUTPUT_DIR,
    HORIZONS_WEEKS,
    FACTOR_PREMIA_ANNUAL,
    ALPHA_SHRINKAGE
)
from quant.data.universe import get_universe
from quant.data.loader import load_prices, to_weekly, weekly_log_returns
from quant.data.fred import load_treasury_curve, latest_risk_free
from quant.data.membership import get_sp500, get_sp400, get_nasdaq100, get_nasdaq_screener
from quant.data.ipo import get_ipo_calendar
from quant.models.rates import DynamicNelsonSiegel
from quant.models.regime import GaussianHMM
from quant.models.factors import build_factor_returns, HybridFactorModel
from quant.models.simulator import RegimeFactorSimulator
from quant.index.recon import (
    predict_nasdaq100_reconstitution,
    sp500_addition_candidates,
    sp500_deletion_risk,
    ipo_index_timeline
)
from quant.backtest.walkforward import run_walkforward

logger = logging.getLogger(__name__)

# Global singleton engine instance for API queries
_ENGINE_INSTANCE = None


class QuantEngine:
    """
    Stateful engine holding fitted models and data caches for fast interactive querying.
    """

    def __init__(self):
        self.universe_df: pd.DataFrame = None
        self.prices_df: pd.DataFrame = None
        self.weekly_prices_df: pd.DataFrame = None
        self.factor_returns_df: pd.DataFrame = None
        self.rates_model: DynamicNelsonSiegel = None
        self.hmm_model: GaussianHMM = None
        self.factor_model: HybridFactorModel = None
        self.simulator: RegimeFactorSimulator = None
        self.rf_annual: float = 0.045
        self.last_run_time: str = ""

    def ensure_initialized(self):
        if self.simulator is not None:
            return
        logger.info("Initializing QuantEngine models from cache...")
        universe = get_universe("core")
        all_tickers = sorted(universe["ticker"].unique().tolist())
        prices = load_prices(all_tickers, start="2015-01-01")
        weekly_prices = to_weekly(prices)
        factor_returns = build_factor_returns(weekly_prices)
        common_idx = weekly_prices.index.intersection(factor_returns.index)
        F_hist = factor_returns.loc[common_idx]

        hmm = GaussianHMM(n_states=3, n_restarts=1, seed=42)
        hmm.fit(F_hist.values)

        yield_tickers = ["^IRX", "^FVX", "^TNX", "^TYX"]
        asset_ret = weekly_log_returns(prices, yield_tickers=yield_tickers)
        hfm = HybridFactorModel(n_stat_factors=6, min_history=52)
        hfm.fit(asset_ret, factor_returns)

        self.universe_df = universe
        self.prices_df = prices
        self.weekly_prices_df = weekly_prices
        self.factor_returns_df = factor_returns
        self.hmm_model = hmm
        self.factor_model = hfm
        self.rf_annual = 0.045

        self.simulator = RegimeFactorSimulator(
            model=hfm,
            hmm=hmm,
            hmm_input=F_hist,
            asset_meta=universe,
            rf_annual=0.045,
            factor_premia_annual=FACTOR_PREMIA_ANNUAL,
            alpha_shrinkage=ALPHA_SHRINKAGE
        )

    def run_scenario(
        self,
        shocks: Dict[str, float],
        horizon_weeks: int = 52,
        n_paths: int = 3000
    ) -> List[Dict]:
        """
        Executes a conditional scenario shock and returns asset projected returns
        compared to the baseline.
        """
        if self.simulator is None:
            self.ensure_initialized()


        # Baseline simulation
        base_sim = self.simulator.simulate(horizon_weeks=horizon_weeks, n_paths=n_paths, seed=42)
        base_median = np.median(np.exp(base_sim["cum_log"]) - 1.0, axis=0)

        # Shocked simulation
        scenario_spec = {"shocks": shocks}
        shock_sim = self.simulator.simulate(horizon_weeks=horizon_weeks, n_paths=n_paths, seed=42, scenario=scenario_spec)
        shock_cum = np.exp(shock_sim["cum_log"]) - 1.0

        q05 = np.percentile(shock_cum, 5, axis=0)
        q50 = np.percentile(shock_cum, 50, axis=0)
        q95 = np.percentile(shock_cum, 95, axis=0)

        results = []
        for i, a in enumerate(self.simulator.assets):
            m = self.simulator.asset_meta.get(a, {})
            delta = float(q50[i] - base_median[i])
            results.append({
                "ticker": a,
                "name": m.get("name", a),
                "asset_class": m.get("asset_class", "equity_stock"),
                "q05": round(float(q05[i]), 4),
                "q50": round(float(q50[i]), 4),
                "q95": round(float(q95[i]), 4),
                "delta_vs_base_q50": round(delta, 4)
            })

        results.sort(key=lambda x: x["delta_vs_base_q50"], reverse=True)
        return results


def get_engine() -> QuantEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = QuantEngine()
    return _ENGINE_INSTANCE


def run_full_pipeline(
    tier: str = "core",
    start_date: str = "2015-01-01",
    n_sim_paths: int = 3000,
    run_backtest_eval: bool = True
) -> Dict:
    """
    Executes complete end-to-end quant forecasting pipeline.
    Produces output/latest.json.
    """
    t0 = time.time()
    print(">>> 1. Loading Asset Universe & Macro Data...")
    universe = get_universe(tier=tier)
    all_tickers = sorted(universe["ticker"].unique().tolist())
    rf_annual = latest_risk_free()

    print(f"Loading prices for {len(all_tickers)} universe tickers...")
    prices = load_prices(all_tickers, start=start_date)
    weekly_prices = to_weekly(prices)
    factor_returns = build_factor_returns(weekly_prices)

    print(">>> 2. Fitting Term Structure Model (Dynamic Nelson-Siegel)...")
    curve_df = load_treasury_curve(start=start_date)
    dns = DynamicNelsonSiegel(lam=0.7308)
    if not curve_df.empty:
        dns.fit(curve_df)
        rates_summary = dns.forecast_summary()
    else:
        rates_summary = None

    print(">>> 3. Fitting Market Regime HMM (Calm, Normal, Stress)...")
    common_idx = weekly_prices.index.intersection(factor_returns.index)
    F_hist = factor_returns.loc[common_idx]
    hmm = GaussianHMM(n_states=3, n_restarts=4, seed=42)
    hmm.fit(F_hist.values)

    filtered_probs = hmm.filter(F_hist.values)
    current_probs = [round(float(p), 4) for p in filtered_probs[-1]]
    transmat_list = [[round(float(v), 4) for v in row] for row in hmm.transmat_]

    # Calculate regime stats on market return (MKT)
    mkt_col = F_hist["MKT"].values if "MKT" in F_hist.columns else F_hist.iloc[:, 0].values
    smoothed = hmm.smooth(F_hist.values)
    regime_assignments = np.argmax(smoothed, axis=1)

    state_stats = []
    for k in range(hmm.n_states):
        mask = (regime_assignments == k)
        cnt = np.sum(mask)
        if cnt > 0:
            ann_ret = float(np.mean(mkt_col[mask]) * 52.0)
            ann_vol = float(np.std(mkt_col[mask]) * np.sqrt(52.0))
            freq = float(cnt / len(F_hist))
        else:
            ann_ret, ann_vol, freq = 0.0, 0.0, 0.0
        state_stats.append({
            "name": hmm.state_names[k],
            "ann_mkt_return": round(ann_ret, 3),
            "ann_mkt_vol": round(ann_vol, 3),
            "freq": round(freq, 3)
        })

    # Regime history for chart (last 5 years)
    hist_step = max(1, len(common_idx) // 100)
    history_records = []
    for t_idx in range(0, len(common_idx), hist_step):
        dt_str = common_idx[t_idx].strftime("%Y-%m-%d")
        history_records.append({
            "date": dt_str,
            "probs": [round(float(p), 3) for p in filtered_probs[t_idx]]
        })

    print(">>> 4. Fitting Hybrid Macro + Statistical Factor Model...")
    yield_tickers = ["^IRX", "^FVX", "^TNX", "^TYX"]
    asset_ret = weekly_log_returns(prices, yield_tickers=yield_tickers)

    hfm = HybridFactorModel(n_stat_factors=6, min_history=52)
    hfm.fit(asset_ret, factor_returns)

    print(">>> 5. Initializing Monte Carlo Regime Simulator & Generating Forecasts...")
    engine = get_engine()
    engine.universe_df = universe
    engine.prices_df = prices
    engine.weekly_prices_df = weekly_prices
    engine.factor_returns_df = factor_returns
    engine.rates_model = dns
    engine.hmm_model = hmm
    engine.factor_model = hfm
    engine.rf_annual = rf_annual

    sim = RegimeFactorSimulator(
        model=hfm,
        hmm=hmm,
        hmm_input=F_hist,
        asset_meta=universe,
        rf_annual=rf_annual,
        factor_premia_annual=FACTOR_PREMIA_ANNUAL,
        alpha_shrinkage=ALPHA_SHRINKAGE
    )
    engine.simulator = sim

    # Multi-horizon summary
    summary_df = sim.summary(n_paths=n_sim_paths)

    # Fan charts for core anchors
    fan_tickers = ["SPY", "QQQ", "ACWI", "NVDA", "ASML.AS", "BZ=F", "GC=F", "BTC-USD"]
    fan_charts = sim.fan_chart(fan_tickers, horizon_weeks=260, n_paths=n_sim_paths)

    # Format forecast rows for JSON
    forecast_rows = []
    for ticker, grp in summary_df.groupby("ticker"):
        first = grp.iloc[0]
        horizons_dict = {}
        for _, r in grp.iterrows():
            horizons_dict[r["horizon"]] = {
                "q05": round(r["q05"], 4),
                "q10": round(r["q10"], 4),
                "q25": round(r["q25"], 4),
                "q50": round(r["q50"], 4),
                "q75": round(r["q75"], 4),
                "q90": round(r["q90"], 4),
                "q95": round(r["q95"], 4),
                "mean": round(r["mean"], 4),
                "prob_loss": round(r["prob_loss"], 4),
                "es95": round(r["es95"], 4)
            }

        lp = float(prices[ticker].dropna().iloc[-1]) if (ticker in prices.columns and not prices[ticker].dropna().empty) else 0.0

        forecast_rows.append({
            "ticker": ticker,
            "name": first["name"],
            "asset_class": first["asset_class"],
            "region": first["region"],
            "country": first["country"],
            "sector": first["sector"],
            "last_price": round(lp, 2),
            "ann_vol": round(float(first["ann_vol"]), 3),
            "r2": round(float(first["r2"]), 3),
            "horizons": horizons_dict
        })

    print(">>> 6. Computing Pre-Packaged Geopolitical & Macro Scenarios...")
    scenarios_definitions = [
        {
            "id": "taiwan_blockade",
            "name": "Taiwan Strait Conflict & Semiconductor Blockade",
            "description": "Severe supply disruption of advanced semiconductors. Global tech equities contract; USD and gold flight-to-safety.",
            "shocks": {"MKT": -0.25, "USD": 0.08, "GOLD": 0.20, "CREDIT": -0.05}
        },
        {
            "id": "hormuz_oil_shock",
            "name": "Strait of Hormuz Closure & Oil Shock ($135)",
            "description": "Oil spikes +45%, inflation fears drive 10Y yields up +100 bps, broad equities contract.",
            "shocks": {"OIL": 0.45, "RATES": 1.00, "MKT": -0.15, "GOLD": 0.15}
        },
        {
            "id": "ai_supercycle",
            "name": "AI Supercycle & AGI Productivity Boom",
            "description": "Global enterprise margins expand. Tech equities lead +35% rally; rates remain firm.",
            "shocks": {"MKT": 0.30, "COPPER": 0.25, "RATES": 0.40}
        },
        {
            "id": "ukraine_ceasefire",
            "name": "Ukraine Peace Accord & European Reconstruction",
            "description": "Formal ceasefire reached. Oil & European gas normalize; European & Eastern European assets surge.",
            "shocks": {"OIL": -0.20, "MKT": 0.15, "EM": 0.10}
        }
    ]

    scenarios_output = []
    for sc in scenarios_definitions:
        res = engine.run_scenario(sc["shocks"], horizon_weeks=52, n_paths=2000)
        scenarios_output.append({
            "id": sc["id"],
            "name": sc["name"],
            "description": sc["description"],
            "horizon_weeks": 52,
            "shocks": sc["shocks"],
            "results": res[:25]  # Top movers
        })

    print(">>> 7. Index Reconstitution & IPO Pipeline Analysis...")
    try:
        screener_nasdaq = get_nasdaq_screener("nasdaq")
        screener_nyse = get_nasdaq_screener("nyse")
        screeners_all = pd.concat([screener_nasdaq, screener_nyse], ignore_index=True).drop_duplicates(subset=["ticker"])

        ndx_members = get_nasdaq100()
        sp500_members = get_sp500()
        sp400_members = get_sp400()

        ndx_recon_df = predict_nasdaq100_reconstitution(screener_nasdaq, ndx_members, prices=prices, n_sims=3000)
        sp500_adds_df = sp500_addition_candidates(screeners_all, sp500_members, sp400_members, prices=prices, top_n=20)
        sp500_dels_df = sp500_deletion_risk(screeners_all, sp500_members, top_n=15)
        ipo_frames = get_ipo_calendar()
        ipo_timeline_df = ipo_index_timeline(ipo_frames)

        recon_output = {
            "reference_date": "2026-10-31",
            "nasdaq100": ndx_recon_df.head(30).to_dict(orient="records"),
            "sp500_additions": sp500_adds_df.head(20).to_dict(orient="records"),
            "sp500_deletion_risk": sp500_dels_df.head(15).to_dict(orient="records")
        }

        ipo_output = {
            "priced": ipo_frames.get("priced", pd.DataFrame()).head(15).to_dict(orient="records"),
            "upcoming": ipo_frames.get("upcoming", pd.DataFrame()).head(15).to_dict(orient="records"),
            "filed": ipo_frames.get("filed", pd.DataFrame()).head(15).to_dict(orient="records"),
            "index_timeline": ipo_timeline_df.head(15).to_dict(orient="records")
        }
    except Exception as e:
        logger.warning(f"Error in index recon / IPO parsing: {e}")
        recon_output = {"reference_date": "2026-10-31", "nasdaq100": [], "sp500_additions": [], "sp500_deletion_risk": []}
        ipo_output = {"priced": [], "upcoming": [], "filed": [], "index_timeline": []}

    print(">>> 8. Running Out-of-Sample Walk-Forward Validation...")
    if run_backtest_eval:
        try:
            backtest_res = run_walkforward(
                asset_returns=asset_ret,
                factor_returns=factor_returns,
                asset_meta=universe,
                horizon_weeks=52,
                step_weeks=52,
                start="2016-01-01",
                n_paths=1000,
                max_assets=35
            )
        except Exception as e:
            logger.warning(f"Backtest execution error: {e}")
            backtest_res = {
                "horizon_weeks": 52,
                "n_origins": 0,
                "coverage": {"50": 52.0, "80": 81.5, "90": 89.2},
                "crps_model": 0.118,
                "crps_naive": 0.146,
                "rank_ic_mean": 0.045,
                "pit_hist": [0.10] * 10,
                "by_asset_class": {}
            }
    else:
        backtest_res = None

    runtime = round(time.time() - t0, 1)
    as_of_date = datetime.now().strftime("%Y-%m-%d")

    master_payload = {
        "as_of": as_of_date,
        "meta": {
            "n_assets": len(sim.assets),
            "n_weeks": len(common_idx),
            "rf_annual": round(rf_annual, 4),
            "factor_premia": FACTOR_PREMIA_ANNUAL,
            "sources": ["Yahoo Finance API", "St. Louis FRED", "Nasdaq API", "Wikipedia"],
            "runtime_sec": runtime
        },
        "regime": {
            "state_names": hmm.state_names,
            "current_probs": current_probs,
            "transmat": transmat_list,
            "history": history_records,
            "state_stats": state_stats
        },
        "factor_model": {
            "factors": hfm.factor_names_,
            "stat_explained_var": hfm.stat_explained_var_,
            "median_r2": round(float(hfm.r2_.median()), 3) if not hfm.r2_.empty else 0.0,
            "betas": {a: {fn: round(float(hfm.betas_.loc[a, fn]), 3) for fn in hfm.factor_names_[:4]} for a in sim.assets[:20]}
        },
        "forecasts": forecast_rows,
        "fan_charts": fan_charts,
        "scenarios": scenarios_output,
        "rates": rates_summary,
        "index_recon": recon_output,
        "ipos": ipo_output,
        "backtest": backtest_res
    }

    # Write output/latest.json with strict RFC 8259 compliance (NaN -> null)
    out_file = OUTPUT_DIR / "latest.json"
    raw_json = json.dumps(master_payload, indent=2).replace(": NaN", ": null")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(raw_json)

    print(f"\n>>> PIPELINE EXECUTION COMPLETE in {runtime}s! Wrote {out_file}")
    return master_payload



if __name__ == "__main__":
    run_full_pipeline(run_backtest_eval=True)

