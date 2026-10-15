"""
titan.backtest.walkforward — Strict Out-of-Sample Walk-Forward Validation Engine.
Evaluates forecasting calibration without look-ahead bias:
- Empirical Coverage (50%, 80%, 90% confidence bands)
- Probability Integral Transform (PIT) histogram
- Continuous Ranked Probability Score (CRPS) vs Naive Gaussian Random Walk Benchmark
- Cross-sectional Spearman Rank Information Coefficient (IC)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from titan.config import (
    BACKTEST_HORIZON_WEEKS,
    BACKTEST_MAX_ASSETS,
    BACKTEST_N_PATHS,
    BACKTEST_START,
    BACKTEST_STEP_WEEKS,
    ESTIMATION_WINDOW_WEEKS,
    N_REGIMES,
)
from titan.models.factors import HybridFactorModel
from titan.models.regime import GaussianHMM
from titan.sim.simulator import RegimeFactorSimulator

logger = logging.getLogger(__name__)


def _compute_crps_ensemble(sim_draws: np.ndarray, y: float) -> float:
    m = len(sim_draws)
    if m == 0:
        return 0.0
    x_sorted = np.sort(sim_draws)
    term1 = np.mean(np.abs(x_sorted - y))
    weights = 2 * np.arange(1, m + 1) - m - 1
    term2 = np.sum(weights * x_sorted) / (m * m)
    return float(term1 - term2)


def _compute_crps_gaussian(mu: float, sigma: float, y: float) -> float:
    if sigma <= 1e-8:
        return float(abs(y - mu))
    z = (y - mu) / sigma
    phi = norm.pdf(z)
    big_phi = norm.cdf(z)
    crps = sigma * (z * (2 * big_phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))
    return float(crps)


def run_walkforward(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    asset_meta: pd.DataFrame,
    horizon_weeks: int = BACKTEST_HORIZON_WEEKS,
    step_weeks: int = BACKTEST_STEP_WEEKS,
    start: str = BACKTEST_START,
    n_paths: int = BACKTEST_N_PATHS,
    seed: int = 42,
    window_weeks: int = ESTIMATION_WINDOW_WEEKS,
    n_states: int = N_REGIMES,
    max_assets: int = BACKTEST_MAX_ASSETS,
) -> dict[str, Any]:
    """
    Executes walk-forward backtest across historical origins strictly without look-ahead bias.
    """
    common_idx = asset_returns.index.intersection(factor_returns.index).sort_values()
    start_dt = pd.to_datetime(start)

    coverage = asset_returns.loc[common_idx].notna().sum()
    top_assets = coverage.sort_values(ascending=False).head(max_assets).index.tolist()

    y_all = asset_returns.loc[common_idx, top_assets]
    f_all = factor_returns.loc[common_idx]

    eligible_dates = [d for d in common_idx if d >= start_dt]
    t_total = len(common_idx)
    origin_indices = [
        common_idx.get_loc(d) for d in eligible_dates if common_idx.get_loc(d) + horizon_weeks < t_total
    ]

    sampled_origins = origin_indices[::step_weeks]
    if not sampled_origins and origin_indices:
        sampled_origins = [origin_indices[0]]

    pit_values: list[float] = []
    cov50_hits: list[float] = []
    cov80_hits: list[float] = []
    cov90_hits: list[float] = []
    crps_models: list[float] = []
    crps_naives: list[float] = []
    rank_ics: list[float] = []
    per_origin_records: list[dict] = []

    meta_dict = (
        asset_meta.set_index("ticker").to_dict(orient="index")
        if "ticker" in asset_meta.columns
        else {}
    )

    for pos in sampled_origins:
        origin_dt = common_idx[pos]
        start_pos = max(0, pos - window_weeks)
        window_idx = common_idx[start_pos : pos + 1]

        y_hist = y_all.loc[window_idx]
        f_hist = f_all.loc[window_idx]

        future_idx = common_idx[pos + 1 : pos + 1 + horizon_weeks]
        y_future = y_all.loc[future_idx]
        realized = y_future.sum(axis=0)

        # Assets with at least 52 observations in the past window
        valid_eval_assets = [a for a in top_assets if y_hist[a].notna().sum() >= 52]
        if len(valid_eval_assets) < 5:
            continue

        try:
            model = HybridFactorModel(n_stat_factors=3, min_history=52)
            model.fit(y_hist[valid_eval_assets], f_hist)

            hmm = GaussianHMM(n_states=n_states, n_restarts=2, seed=seed)
            hmm.fit(f_hist.values)

            sim = RegimeFactorSimulator(
                model=model,
                hmm=hmm,
                hmm_input=f_hist,
                asset_meta=asset_meta,
                rf_annual=0.03,
            )

            sim_out = sim.simulate(horizon_weeks=horizon_weeks, n_paths=n_paths, seed=seed)
            cum_log_draws = sim_out["cum_log"]

            sim_assets = sim.assets
            eval_assets = [a for a in valid_eval_assets if a in sim_assets]

            origin_model_crps: list[float] = []
            origin_naive_crps: list[float] = []
            f_means: list[float] = []
            r_actuals: list[float] = []

            for a in eval_assets:
                idx_sim = sim_assets.index(a)
                draws = cum_log_draws[:, idx_sim]
                y_act = float(realized[a])

                q05, q10, q25, q75, q90, q95 = np.percentile(draws, [5, 10, 25, 75, 90, 95])
                cov50_hits.append(1.0 if (q25 <= y_act <= q75) else 0.0)
                cov80_hits.append(1.0 if (q10 <= y_act <= q90) else 0.0)
                cov90_hits.append(1.0 if (q05 <= y_act <= q95) else 0.0)

                pit = float(np.mean(draws <= y_act))
                pit_values.append(pit)

                c_model = _compute_crps_ensemble(draws, y_act)
                origin_model_crps.append(c_model)

                hist_vol = (
                    float(np.std(y_hist[a].dropna()) * np.sqrt(horizon_weeks))
                    if len(y_hist[a].dropna()) > 10
                    else 0.15
                )
                c_naive = _compute_crps_gaussian(mu=0.0, sigma=hist_vol, y=y_act)
                origin_naive_crps.append(c_naive)

                f_means.append(float(np.mean(draws)))
                r_actuals.append(y_act)

            if len(f_means) > 5:
                ic_val, _ = spearmanr(f_means, r_actuals)
                if not np.isnan(ic_val):
                    rank_ics.append(float(ic_val))

            mean_cm = float(np.mean(origin_model_crps))
            mean_cn = float(np.mean(origin_naive_crps))
            crps_models.extend(origin_model_crps)
            crps_naives.extend(origin_naive_crps)

            per_origin_records.append({
                "date": origin_dt.strftime("%Y-%m-%d"),
                "n_assets": len(eval_assets),
                "crps_model": round(mean_cm, 4),
                "crps_naive": round(mean_cn, 4),
            })
        except Exception as e:
            logger.debug("Walkforward origin %s failed: %s", origin_dt, e)
            continue

    pit_arr = np.array(pit_values) if pit_values else np.array([0.5])
    counts, _ = np.histogram(pit_arr, bins=10, range=(0.0, 1.0))
    pit_hist = [round(float(c / len(pit_arr)), 3) for c in counts]

    return {
        "horizon_weeks": horizon_weeks,
        "n_origins": len(per_origin_records),
        "coverage": {
            "50": round(float(np.mean(cov50_hits)) * 100.0, 1) if cov50_hits else None,
            "80": round(float(np.mean(cov80_hits)) * 100.0, 1) if cov80_hits else None,
            "90": round(float(np.mean(cov90_hits)) * 100.0, 1) if cov90_hits else None,
        },
        "crps_model": round(float(np.mean(crps_models)), 4) if crps_models else None,
        "crps_naive": round(float(np.mean(crps_naives)), 4) if crps_naives else None,
        "rank_ic_mean": round(float(np.mean(rank_ics)), 3) if rank_ics else None,
        "pit_hist": pit_hist,
        "per_origin": per_origin_records,
    }
