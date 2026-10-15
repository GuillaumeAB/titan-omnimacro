"""
titan.models.rates — Dynamic Nelson-Siegel yield curve model (Diebold & Li 2006).
Decomposes term structure into Level, Slope, and Curvature factors,
fits VAR(1) dynamics with stationarity projection, and simulates yield paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from titan.config import DNS_LAMBDA_MONTHLY, RATES_HORIZONS_MONTHS


class DynamicNelsonSiegel:
    """
    Dynamic Nelson-Siegel (Diebold-Li 2006) yield curve model.
    Maturities tau in years.
    """

    def __init__(self, lam: float = 0.7308):
        self.lam = lam
        self.maturities_: np.ndarray | None = None
        self.factors_: pd.DataFrame | None = None
        self.var_A_: np.ndarray | None = None
        self.var_c_: np.ndarray | None = None
        self.residuals_: np.ndarray | None = None
        self.fit_rmse_bp: float = 0.0
        self.current_curve_: pd.Series | None = None

    def _nelson_siegel_loadings(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=float)
        l_tau = np.maximum(self.lam * tau, 1e-6)

        factor1 = np.ones_like(tau)
        factor2 = (1.0 - np.exp(-l_tau)) / l_tau
        factor3 = factor2 - np.exp(-l_tau)
        return np.column_stack([factor1, factor2, factor3])

    def fit(self, yields: pd.DataFrame) -> DynamicNelsonSiegel:
        df = yields.dropna(how="all").ffill().bfill()
        if df.empty:
            raise ValueError("Yields DataFrame is empty")

        maturities = np.array([float(c) for c in df.columns])
        self.maturities_ = maturities
        self.current_curve_ = df.iloc[-1]

        h = self._nelson_siegel_loadings(maturities)
        y = df.values

        h_inv = np.linalg.pinv(h.T @ h) @ h.T
        betas = (h_inv @ y.T).T

        self.factors_ = pd.DataFrame(betas, index=df.index, columns=["Level", "Slope", "Curvature"])

        y_fitted = betas @ h.T
        rmse_pct = np.sqrt(np.mean((y - y_fitted) ** 2))
        self.fit_rmse_bp = float(rmse_pct * 100.0)

        # VAR(1) fit on factors: B_t = c + B_{t-1} @ A.T + e_t
        t = len(betas)
        if t > 2:
            y_var = betas[1:]
            x_var = np.column_stack([np.ones(t - 1), betas[:-1]])
            coeff = np.linalg.pinv(x_var.T @ x_var) @ (x_var.T @ y_var)
            c = coeff[0]
            a = coeff[1:].T

            eigvals = np.linalg.eigvals(a)
            max_eig = np.max(np.abs(eigvals))
            if max_eig >= 0.999:
                a = a * (0.98 / max_eig)

            self.var_c_ = c
            self.var_A_ = a
            self.residuals_ = y_var - (x_var @ coeff)
        else:
            self.var_c_ = np.zeros(3)
            self.var_A_ = np.eye(3) * 0.98
            self.residuals_ = np.zeros((1, 3))

        return self

    def simulate(
        self,
        horizon_months: int,
        n_paths: int = 2000,
        seed: int = 42,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        last_beta = self.factors_.iloc[-1].values
        h = self._nelson_siegel_loadings(self.maturities_)

        resids = (
            self.residuals_
            if (self.residuals_ is not None and len(self.residuals_) > 5)
            else rng.normal(0, 0.1, (10, 3))
        )
        n_resids = len(resids)

        beta_paths = np.zeros((n_paths, horizon_months, 3))
        current_b = np.tile(last_beta, (n_paths, 1))

        for m in range(horizon_months):
            idx = rng.choice(n_resids, size=n_paths)
            innovations = resids[idx]
            next_b = self.var_c_ + (current_b @ self.var_A_.T) + innovations
            beta_paths[:, m, :] = next_b
            current_b = next_b

        return np.matmul(beta_paths, h.T)

    def forecast_summary(
        self,
        horizons_months: dict[str, int] | None = None,
        quantiles: list[float] | None = None,
    ) -> dict:
        if horizons_months is None:
            horizons_months = RATES_HORIZONS_MONTHS
        if quantiles is None:
            quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        max_h = max(horizons_months.values())
        sim_paths = self.simulate(horizon_months=max_h, n_paths=2500)

        summary_horizons = {}
        maturities_list = [float(m) for m in self.maturities_]

        for h_name, h_mo in horizons_months.items():
            step = min(h_mo - 1, max_h - 1)
            slice_y = sim_paths[:, step, :]

            q_dict = {
                f"q{int(q*100):02d}": [round(float(v), 3) for v in np.percentile(slice_y, q * 100, axis=0)]
                for q in quantiles
            }
            q_dict["mean"] = [round(float(v), 3) for v in np.mean(slice_y, axis=0)]
            summary_horizons[h_name] = q_dict

        current_vals = [round(float(self.current_curve_.iloc[i]), 3) for i in range(len(self.maturities_))]

        return {
            "maturities": maturities_list,
            "current": current_vals,
            "horizons": summary_horizons,
            "fit_rmse_bp": round(self.fit_rmse_bp, 2),
        }
