"""
quant.models.simulator — Regime-Conditioned Factor Monte Carlo Simulator.
Generates multi-horizon joint return distributions across hundreds of global assets
using Markov regime transitions, factor bootstrap with risk premia, residual resampling,
and Gaussian conditioning for custom macro/geopolitical shock scenarios.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from quant.config import HORIZONS_WEEKS, FACTOR_PREMIA_ANNUAL, ALPHA_SHRINKAGE
from quant.models.factors import HybridFactorModel
from quant.models.regime import GaussianHMM


class RegimeFactorSimulator:
    """
    Monte Carlo multi-horizon path generator conditioned on market regimes and factor dynamics.
    """

    def __init__(
        self,
        model: HybridFactorModel,
        hmm: GaussianHMM,
        hmm_input: pd.DataFrame,
        asset_meta: pd.DataFrame,
        rf_annual: float = 0.045,
        factor_premia_annual: Dict[str, float] = None,
        alpha_shrinkage: float = ALPHA_SHRINKAGE
    ):
        self.model = model
        self.hmm = hmm
        self.hmm_input = hmm_input
        self.rf_annual = rf_annual
        self.factor_premia_annual = factor_premia_annual or FACTOR_PREMIA_ANNUAL
        self.alpha_shrinkage = alpha_shrinkage

        # Filter out rates_yield assets from asset simulation universe
        meta_dict = asset_meta.set_index("ticker").to_dict(orient="index") if "ticker" in asset_meta.columns else {}
        self.asset_meta = meta_dict

        sim_assets = []
        for a in model.assets_:
            m = meta_dict.get(a, {})
            if m.get("asset_class") != "rates_yield":
                sim_assets.append(a)

        self.assets = sim_assets
        self.N = len(sim_assets)
        self.betas = model.betas_.loc[sim_assets].values  # (N, F)
        self.factor_names = model.factor_names_
        self.F = len(self.factor_names)

        # Carry drift per asset (weekly)
        rf_weekly = rf_annual / 52.0
        self.asset_drifts = np.zeros(self.N)

        for i, a in enumerate(sim_assets):
            m = meta_dict.get(a, {})
            ac = m.get("asset_class", "equity_stock")
            # Carry rule
            if ac in ["equity_stock", "equity_index", "equity_country", "equity_sector", "bond", "crypto"]:
                carry = rf_weekly
            else:
                carry = 0.0  # commodity, fx

            # Historical excess alpha
            hist_alpha = float(model.alpha_.get(a, 0.0))
            alpha_excess = hist_alpha - carry
            self.asset_drifts[i] = carry + (self.alpha_shrinkage * alpha_excess)

        # Pool factor returns by smoothed HMM state
        common_idx = model.factor_returns_.index.intersection(hmm_input.index)
        F_hist = model.factor_returns_.loc[common_idx].values  # (T, F)
        smoothed = hmm.smooth(hmm_input.loc[common_idx].values)  # (T, K)
        states = np.argmax(smoothed, axis=1)

        self.state_factor_pools: Dict[int, np.ndarray] = {}
        for k in range(hmm.n_states):
            mask = (states == k)
            if np.any(mask):
                self.state_factor_pools[k] = F_hist[mask]
            else:
                self.state_factor_pools[k] = F_hist

        # Weekly forward premia vector
        self.weekly_premia = np.zeros(self.F)
        uncond_mean = np.mean(F_hist, axis=0)

        for j, fn in enumerate(self.factor_names):
            if fn in self.factor_premia_annual:
                self.weekly_premia[j] = self.factor_premia_annual[fn] / 52.0
            else:
                self.weekly_premia[j] = 0.0

        self.uncond_mean = uncond_mean

        # Residuals pool per asset
        resids_df = model.resid_[sim_assets].dropna(how="all")
        self.resids_pool = {}
        for a in sim_assets:
            s = resids_df[a].dropna().values
            self.resids_pool[a] = s if len(s) > 10 else None

        # Last filtered state probabilities
        filtered = hmm.filter(hmm_input.values)
        self.current_state_probs = filtered[-1]

    def simulate(
        self,
        horizon_weeks: int,
        n_paths: int = 5000,
        seed: int = 42,
        scenario: Optional[Dict] = None,
        path_tickers: Optional[List[str]] = None
    ) -> Dict:
        """
        Simulates cumulative return distributions over the specified horizon.
        """
        rng = np.random.default_rng(seed)
        K = self.hmm.n_states
        trans = self.hmm.transmat_

        # 1. Simulate Markov regime sequences: shape (n_paths, horizon_weeks)
        regimes = np.zeros((n_paths, horizon_weeks), dtype=int)
        # Initial regime from current filtered probabilities
        regimes[:, 0] = rng.choice(K, size=n_paths, p=self.current_state_probs)

        for t in range(1, horizon_weeks):
            prev_s = regimes[:, t - 1]
            for k in range(K):
                mask = (prev_s == k)
                if np.any(mask):
                    regimes[mask, t] = rng.choice(K, size=np.sum(mask), p=trans[k])

        # 2. Draw factor returns for each path and week
        factor_draws = np.zeros((n_paths, horizon_weeks, self.F))
        for k in range(K):
            pool_k = self.state_factor_pools[k]
            n_pool = len(pool_k)
            mask = (regimes == k)
            count = np.sum(mask)
            if count > 0:
                idx = rng.choice(n_pool, size=count)
                factor_draws[mask] = pool_k[idx]

        # Re-center factor draws: subtract unconditional historical mean, add forward premium
        factor_draws = factor_draws - self.uncond_mean + self.weekly_premia

        # Cumulative factor sums: shape (n_paths, F)
        factor_cum = np.sum(factor_draws, axis=1)

        # 3. Apply Gaussian conditioning if scenario shocks are specified
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

                # Sample covariance of simulated factor sums
                cov_factors = np.cov(factor_cum, rowvar=False)
                mean_factors = np.mean(factor_cum, axis=0)

                Sigma_cc = cov_factors[np.ix_(c_idx, c_idx)] + np.eye(len(c_idx)) * 1e-6
                Sigma_fc = cov_factors[np.ix_(f_idx, c_idx)]
                inv_Sigma_cc = np.linalg.pinv(Sigma_cc)

                delta_c = np.array(target_vals) - factor_cum[:, c_idx]  # (n_paths, C)

                if len(f_idx) > 0:
                    adjust_f = (Sigma_fc @ inv_Sigma_cc @ delta_c.T).T  # (n_paths, Free)
                    factor_cum[:, f_idx] += adjust_f

                factor_cum[:, c_idx] = np.tile(target_vals, (n_paths, 1))

        # 4. Compute cumulative asset returns = drift * H + beta * factor_cum + cum_resids
        # Cumulative drift: (N,)
        cum_drift = self.asset_drifts * horizon_weeks

        # Factor contribution: (n_paths, N) = factor_cum @ betas^T
        factor_contrib = factor_cum @ self.betas.T

        # Residuals bootstrap (sum of H iid weekly draws)
        resid_contrib = np.zeros((n_paths, self.N))
        for i, a in enumerate(self.assets):
            pool = self.resids_pool.get(a)
            if pool is not None and len(pool) > 10:
                # Sum of H draws for n_paths
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
                    beta_pt = self.betas[idx_pt]  # (F,)
                    drift_pt = self.asset_drifts[idx_pt]

                    # Factor paths: (n_paths, horizon_weeks)
                    factor_step_contrib = factor_draws @ beta_pt  # (n_paths, H)

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
            "factor_cum": factor_cum
        }

    def summary(
        self,
        horizons: Dict[str, int] = None,
        n_paths: int = 3000,
        seed: int = 42,
        scenario: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Generates forecast summary across assets and horizons.
        Outputs simple return quantiles: q05..q95, mean, prob_loss, es95, ann_vol.
        """
        if horizons is None:
            horizons = HORIZONS_WEEKS

        rows = []
        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        for h_name, h_weeks in horizons.items():
            sim_res = self.simulate(horizon_weeks=h_weeks, n_paths=n_paths, seed=seed, scenario=scenario)
            cum_log = sim_res["cum_log"]  # (n_paths, N)
            simple_ret = np.exp(cum_log) - 1.0  # (n_paths, N)

            q_vals = np.percentile(simple_ret, [q * 100 for q in quantiles], axis=0)  # (7, N)
            means = np.mean(simple_ret, axis=0)
            prob_loss = np.mean(simple_ret < 0.0, axis=0)

            # Expected Shortfall (CVaR at 5%)
            q05_thresh = q_vals[0]
            es95 = np.zeros(self.N)
            for i in range(self.N):
                tail = simple_ret[:, i][simple_ret[:, i] <= q05_thresh[i]]
                es95[i] = np.mean(tail) if len(tail) > 0 else q05_thresh[i]

            ann_factor = np.sqrt(52.0 / h_weeks)
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
                    "r2": float(self.model.r2_.get(a, 0.0))
                })

        return pd.DataFrame(rows)

    def fan_chart(
        self,
        tickers: List[str],
        horizon_weeks: int = 260,
        n_paths: int = 2500,
        step: int = 4
    ) -> Dict:
        """
        Constructs fan chart progression data (growth of 1.0) along time steps.
        """
        valid_tickers = [t for t in tickers if t in self.assets]
        if not valid_tickers:
            return {}

        sim_res = self.simulate(
            horizon_weeks=horizon_weeks,
            n_paths=n_paths,
            path_tickers=valid_tickers
        )

        paths = sim_res["paths"]
        fan_data = {}
        weeks_steps = list(range(0, horizon_weeks + 1, step))
        if weeks_steps[-1] != horizon_weeks:
            weeks_steps.append(horizon_weeks)

        for t in valid_tickers:
            w_paths = paths.get(t)  # shape (n_paths, horizon_weeks)
            if w_paths is None:
                continue

            # Convert to growth of 1: exp(log_return)
            growth = np.exp(w_paths)
            growth_with_0 = np.column_stack([np.ones(n_paths), growth])

            sub_steps = [growth_with_0[:, w] for w in weeks_steps]
            stacked = np.array(sub_steps)  # (n_steps, n_paths)

            q05 = [round(float(v), 4) for v in np.percentile(stacked, 5, axis=1)]
            q25 = [round(float(v), 4) for v in np.percentile(stacked, 25, axis=1)]
            q50 = [round(float(v), 4) for v in np.percentile(stacked, 50, axis=1)]
            q75 = [round(float(v), 4) for v in np.percentile(stacked, 75, axis=1)]
            q95 = [round(float(v), 4) for v in np.percentile(stacked, 95, axis=1)]

            # Compute median maximum drawdown across paths
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
                "max_dd_median": round(median_max_dd, 4)
            }

        return fan_data
