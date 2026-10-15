"""
titan.sim.simulator — Joint Multi-Asset Regime Factor Monte Carlo Simulator.
Generates multi-horizon joint return distributions across global assets using
Markov regime switching, stationary factor bootstrapping with forward premia,
Ito convexity correction, residual resampling, and Gaussian scenario conditioning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from titan.config import (
    CASH_CARRY_CLASSES,
    FACTOR_PREMIA_ANNUAL,
    HORIZONS_WEEKS,
    N_PATHS,
    SEED,
    WEEKS_PER_YEAR,
)
from titan.models.factors import HybridFactorModel
from titan.models.regime import GaussianHMM


class RegimeFactorSimulator:
    """
    Joint multi-horizon path generator conditioned on macro regimes and factor dynamics.
    """

    def __init__(
        self,
        model: HybridFactorModel,
        hmm: GaussianHMM,
        hmm_input: pd.DataFrame,
        asset_meta: pd.DataFrame,
        rf_annual: float = 0.045,
        factor_premia_annual: dict[str, float] | None = None,
    ):
        self.model = model
        self.hmm = hmm
        self.hmm_input = hmm_input
        self.rf_annual = rf_annual
        self.factor_premia_annual = factor_premia_annual or FACTOR_PREMIA_ANNUAL

        meta_dict = (
            asset_meta.set_index("ticker").to_dict(orient="index")
            if "ticker" in asset_meta.columns
            else {}
        )
        self.asset_meta = meta_dict

        # Filter out rates_yield assets from asset simulation universe
        sim_assets = [
            a for a in model.assets_ if meta_dict.get(a, {}).get("asset_class") != "rates_yield"
        ]
        self.assets = sim_assets
        self.N = len(sim_assets)
        self.betas = model.betas_.loc[sim_assets].values  # (N, F)
        self.factor_names = model.factor_names_
        self.F = len(self.factor_names)

        rf_weekly = rf_annual / WEEKS_PER_YEAR
        self.asset_drifts = np.zeros(self.N)

        # Asset carry
        for i, a in enumerate(sim_assets):
            ac = meta_dict.get(a, {}).get("asset_class", "equity_stock")
            carry = rf_weekly if ac in CASH_CARRY_CLASSES else 0.0
            # Under quantitative baseline, historical alpha is shrunken heavily towards zero
            self.asset_drifts[i] = carry

        # Pool historical factor returns by smoothed HMM state
        common_idx = model.factor_returns_.index.intersection(hmm_input.index)
        f_hist = model.factor_returns_.loc[common_idx].values
        smoothed = hmm.smooth(hmm_input.loc[common_idx].values)
        states = np.argmax(smoothed, axis=1)

        self.state_factor_pools: dict[int, np.ndarray] = {}
        for k in range(hmm.n_states):
            mask = states == k
            self.state_factor_pools[k] = f_hist[mask] if np.any(mask) else f_hist

        # Weekly forward premia vector with log-convexity correction: mu_log = mu_arith - 0.5 * sigma^2
        self.weekly_premia = np.zeros(self.F)
        uncond_mean = np.mean(f_hist, axis=0)
        uncond_var = np.var(f_hist, axis=0)

        for j, fn in enumerate(self.factor_names):
            if fn in self.factor_premia_annual:
                arith_prem = self.factor_premia_annual[fn] / WEEKS_PER_YEAR
                # Convexity correction
                log_prem = arith_prem - 0.5 * uncond_var[j]
                self.weekly_premia[j] = log_prem
            else:
                self.weekly_premia[j] = 0.0

        self.uncond_mean = uncond_mean

        # Residuals pool per asset
        resids_df = model.resid_[sim_assets].dropna(how="all")
        self.resids_pool: dict[str, np.ndarray | None] = {}
        for a in sim_assets:
            s = resids_df[a].dropna().values
            self.resids_pool[a] = s if len(s) > 10 else None

        filtered = hmm.filter(hmm_input.values)
        self.current_state_probs = filtered[-1]

    def simulate(
        self,
        horizon_weeks: int,
        n_paths: int = N_PATHS,
        seed: int = SEED,
        scenario: dict | None = None,
        path_tickers: list[str] | None = None,
    ) -> dict:
        rng = np.random.default_rng(seed)
        k = self.hmm.n_states
        trans = self.hmm.transmat_

        # 1. Simulate Markov regime sequences
        regimes = np.zeros((n_paths, horizon_weeks), dtype=int)
        regimes[:, 0] = rng.choice(k, size=n_paths, p=self.current_state_probs)

        for t in range(1, horizon_weeks):
            prev_s = regimes[:, t - 1]
            for step_k in range(k):
                mask = prev_s == step_k
                if np.any(mask):
                    regimes[mask, t] = rng.choice(k, size=np.sum(mask), p=trans[step_k])

        # 2. Draw factor returns for each path and week
        factor_draws = np.zeros((n_paths, horizon_weeks, self.F))
        for step_k in range(k):
            pool_k = self.state_factor_pools[step_k]
            n_pool = len(pool_k)
            mask = regimes == step_k
            count = np.sum(mask)
            if count > 0:
                idx = rng.choice(n_pool, size=count)
                factor_draws[mask] = pool_k[idx]

        # Re-center factor draws
        factor_draws = factor_draws - self.uncond_mean + self.weekly_premia
        factor_cum = np.sum(factor_draws, axis=1)

        # 3. Gaussian conditioning for scenario shocks
        if scenario and "shocks" in scenario and scenario["shocks"]:
            shocks = scenario["shocks"]
            shocked_cols = []
            target_vals = []
            for fn, s_val in shocks.items():
                if fn in self.factor_names:
                    shocked_cols.append(self.factor_names.index(fn))
                    target_vals.append(float(s_val))

            if shocked_cols:
                c_idx = np.array(shocked_cols)
                f_idx = np.array([j for j in range(self.F) if j not in shocked_cols])

                cov_factors = np.cov(factor_cum, rowvar=False)
                sigma_cc = cov_factors[np.ix_(c_idx, c_idx)] + np.eye(len(c_idx)) * 1e-6
                sigma_fc = cov_factors[np.ix_(f_idx, c_idx)]
                inv_sigma_cc = np.linalg.pinv(sigma_cc)

                delta_c = np.array(target_vals) - factor_cum[:, c_idx]

                if len(f_idx) > 0:
                    adjust_f = (sigma_fc @ inv_sigma_cc @ delta_c.T).T
                    factor_cum[:, f_idx] += adjust_f

                factor_cum[:, c_idx] = np.tile(target_vals, (n_paths, 1))

        # 4. Compute cumulative asset returns = drift * H + beta * factor_cum + resids
        cum_drift = self.asset_drifts * horizon_weeks
        factor_contrib = factor_cum @ self.betas.T

        resid_contrib = np.zeros((n_paths, self.N))
        for i, a in enumerate(self.assets):
            pool = self.resids_pool.get(a)
            if pool is not None and len(pool) > 10:
                draws = rng.choice(pool, size=(n_paths, horizon_weeks))
                resid_contrib[:, i] = np.sum(draws, axis=1)
            else:
                vol = float(self.model.resid_vol_.get(a, 0.02))
                resid_contrib[:, i] = rng.normal(0, vol * np.sqrt(horizon_weeks), size=n_paths)

        cum_log_returns = cum_drift + factor_contrib + resid_contrib

        # Path tracking for path_tickers
        path_dict = {}
        if path_tickers:
            for pt in path_tickers:
                if pt in self.assets:
                    idx_pt = self.assets.index(pt)
                    beta_pt = self.betas[idx_pt]
                    drift_pt = self.asset_drifts[idx_pt]

                    factor_step_contrib = factor_draws @ beta_pt
                    pool = self.resids_pool.get(pt)
                    if pool is not None and len(pool) > 10:
                        r_draws = rng.choice(pool, size=(n_paths, horizon_weeks))
                    else:
                        vol = float(self.model.resid_vol_.get(pt, 0.02))
                        r_draws = rng.normal(0, vol, size=(n_paths, horizon_weeks))

                    weekly_cum = np.cumsum(drift_pt + factor_step_contrib + r_draws, axis=1)
                    path_dict[pt] = weekly_cum

        return {
            "cum_log": cum_log_returns,
            "paths": path_dict,
            "factor_cum": factor_cum,
        }

    def summary(
        self,
        horizons: dict[str, int] | None = None,
        n_paths: int = 2500,
        seed: int = SEED,
        scenario: dict | None = None,
    ) -> pd.DataFrame:
        if horizons is None:
            horizons = HORIZONS_WEEKS

        rows = []
        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        for h_name, h_weeks in horizons.items():
            sim_res = self.simulate(horizon_weeks=h_weeks, n_paths=n_paths, seed=seed, scenario=scenario)
            cum_log = sim_res["cum_log"]
            simple_ret = np.exp(cum_log) - 1.0

            q_vals = np.percentile(simple_ret, [q * 100 for q in quantiles], axis=0)
            means = np.mean(simple_ret, axis=0)
            prob_loss = np.mean(simple_ret < 0.0, axis=0)

            q05_thresh = q_vals[0]
            es95 = np.zeros(self.N)
            for i in range(self.N):
                tail = simple_ret[:, i][simple_ret[:, i] <= q05_thresh[i]]
                es95[i] = np.mean(tail) if len(tail) > 0 else q05_thresh[i]

            ann_factor = np.sqrt(WEEKS_PER_YEAR / h_weeks)
            ann_vol = np.std(cum_log, axis=0) * ann_factor

            for i, a in enumerate(self.assets):
                m = self.asset_meta.get(a, {})
                rows.append({
                    "ticker": a,
                    "name": m.get("name", a),
                    "asset_class": m.get("asset_class", "equity_stock"),
                    "region": m.get("region", "Global"),
                    "country": m.get("country", "US"),
                    "sector": m.get("sector", "Blend"),
                    "horizon": h_name,
                    "weeks": h_weeks,
                    "q05": float(q_vals[0, i]),
                    "q10": float(q_vals[1, i]),
                    "q25": float(q_vals[2, i]),
                    "q50": float(q_vals[3, i]),
                    "q75": float(q_vals[4, i]),
                    "q90": float(q_vals[5, i]),
                    "q95": float(q_vals[6, i]),
                    "mean": float(means[i]),
                    "prob_loss": float(prob_loss[i]),
                    "es95": float(es95[i]),
                    "ann_vol": float(ann_vol[i]),
                    "r2": float(self.model.r2_.get(a, 0.0)),
                })

        return pd.DataFrame(rows)

    def fan_chart(
        self,
        tickers: list[str],
        horizon_weeks: int = 260,
        n_paths: int = 2500,
        step: int = 4,
    ) -> dict:
        valid_tickers = [t for t in tickers if t in self.assets]
        if not valid_tickers:
            return {}

        sim_res = self.simulate(horizon_weeks=horizon_weeks, n_paths=n_paths, path_tickers=valid_tickers)
        paths = sim_res["paths"]
        fan_data = {}
        weeks_steps = list(range(0, horizon_weeks + 1, step))
        if weeks_steps[-1] != horizon_weeks:
            weeks_steps.append(horizon_weeks)

        for t in valid_tickers:
            w_paths = paths.get(t)
            if w_paths is None:
                continue

            growth = np.exp(w_paths)
            growth_with_0 = np.column_stack([np.ones(n_paths), growth])

            sub_steps = [growth_with_0[:, w] for w in weeks_steps]
            stacked = np.array(sub_steps)

            q05 = [round(float(v), 4) for v in np.percentile(stacked, 5, axis=1)]
            q25 = [round(float(v), 4) for v in np.percentile(stacked, 25, axis=1)]
            q50 = [round(float(v), 4) for v in np.percentile(stacked, 50, axis=1)]
            q75 = [round(float(v), 4) for v in np.percentile(stacked, 75, axis=1)]
            q95 = [round(float(v), 4) for v in np.percentile(stacked, 95, axis=1)]

            running_max = np.maximum.accumulate(growth_with_0, axis=1)
            drawdowns = (growth_with_0 - running_max) / running_max
            max_dds = np.min(drawdowns, axis=1)
            median_max_dd = float(np.median(max_dds))

            fan_data[t] = {
                "weeks": weeks_steps,
                "q05": q05,
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "q95": q95,
                "max_dd_median": round(median_max_dd, 4),
            }

        return fan_data
