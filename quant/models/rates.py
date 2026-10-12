"""
quant.models.rates — Dynamic Nelson-Siegel yield curve model (Diebold & Li 2006).
Decomposes term structure into Level, Slope, and Curvature factors,
fits VAR(1) dynamics with stationarity projection, and simulates yield paths.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class DynamicNelsonSiegel:
    """
    Dynamic Nelson-Siegel (Diebold-Li 2006) yield curve model.
    Maturities tau in years. Default lambda = 0.7308 (maximizes 2-3Y curvature).
    """

    def __init__(self, lam: float = 0.7308):
        self.lam = lam
        self.maturities_: np.ndarray = None
        self.factors_: pd.DataFrame = None         # (T, 3): beta1 (Level), beta2 (Slope), beta3 (Curvature)
        self.var_A_: np.ndarray = None             # (3, 3) VAR(1) transition matrix
        self.var_c_: np.ndarray = None             # (3,) VAR(1) intercept
        self.residuals_: np.ndarray = None         # (T-1, 3) VAR(1) residuals
        self.fit_rmse_bp: float = 0.0              # In-sample RMSE across all maturities in basis points
        self.current_curve_: pd.Series = None

    def _nelson_siegel_loadings(self, tau: np.ndarray) -> np.ndarray:
        """Computes shape loadings: [1, (1-e^{-lt})/lt, (1-e^{-lt})/lt - e^{-lt}]"""
        tau = np.asarray(tau, dtype=float)
        l_tau = self.lam * tau
        # Avoid division by zero
        l_tau = np.maximum(l_tau, 1e-6)

        factor1 = np.ones_like(tau)
        factor2 = (1.0 - np.exp(-l_tau)) / l_tau
        factor3 = factor2 - np.exp(-l_tau)
        return np.column_stack([factor1, factor2, factor3])

    def fit(self, yields: pd.DataFrame):
        """
        Fits cross-sectional factors for each date, then fits VAR(1) on factor time series.
        yields: DataFrame where columns are float maturities in years and index is DatetimeIndex.
        """
        # Clean yields
        df = yields.dropna(how="all").ffill().bfill()
        if df.empty:
            raise ValueError("Yields DataFrame is empty")

        maturities = np.array([float(c) for c in df.columns])
        self.maturities_ = maturities
        self.current_curve_ = df.iloc[-1]

        H = self._nelson_siegel_loadings(maturities)  # (M, 3)
        Y = df.values                                  # (T, M)

        # Cross-sectional OLS per time step: beta_t = (H' H)^{-1} H' y_t
        H_inv = np.linalg.pinv(H.T @ H) @ H.T          # (3, M)
        betas = (H_inv @ Y.T).T                        # (T, 3)

        self.factors_ = pd.DataFrame(betas, index=df.index, columns=["Level", "Slope", "Curvature"])

        # Compute in-sample fit RMSE in basis points
        Y_fitted = betas @ H.T
        rmse_pct = np.sqrt(np.mean((Y - Y_fitted) ** 2))
        self.fit_rmse_bp = float(rmse_pct * 100.0)     # e.g. 0.05% = 5.0 bp

        # Fit VAR(1) on factors: B_t = c + A * B_{t-1} + e_t
        B = betas
        T = len(B)
        if T > 2:
            Y_var = B[1:]       # (T-1, 3)
            X_var = np.column_stack([np.ones(T - 1), B[:-1]])  # (T-1, 4)
            coeff = np.linalg.pinv(X_var.T @ X_var) @ (X_var.T @ Y_var)  # (4, 3)
            c = coeff[0]
            A = coeff[1:].T     # (3, 3)

            # Check stationarity (spectral radius)
            eigvals = np.linalg.eigvals(A)
            max_eig = np.max(np.abs(eigvals))
            if max_eig >= 0.999:
                A = A * (0.98 / max_eig)

            self.var_c_ = c
            self.var_A_ = A

            # Residuals
            fitted_var = (X_var @ coeff)
            self.residuals_ = Y_var - fitted_var
        else:
            self.var_c_ = np.zeros(3)
            self.var_A_ = np.eye(3) * 0.98
            self.residuals_ = np.zeros((1, 3))

        return self

    def simulate(
        self,
        horizon_months: int,
        n_paths: int = 2000,
        seed: int = 42
    ) -> np.ndarray:
        """
        Simulates yield curves forward using bootstrap of VAR(1) residuals.
        Returns: array of shape (n_paths, horizon_months, n_maturities).
        """
        rng = np.random.default_rng(seed)
        last_beta = self.factors_.iloc[-1].values  # (3,)
        H = self._nelson_siegel_loadings(self.maturities_)  # (M, 3)

        resids = self.residuals_ if (self.residuals_ is not None and len(self.residuals_) > 5) else rng.normal(0, 0.1, (10, 3))
        n_resids = len(resids)

        beta_paths = np.zeros((n_paths, horizon_months, 3))
        current_b = np.tile(last_beta, (n_paths, 1))  # (n_paths, 3)

        for m in range(horizon_months):
            idx = rng.choice(n_resids, size=n_paths)
            innovations = resids[idx]  # (n_paths, 3)

            # B_{t+1} = c + B_t @ A' + e
            next_b = self.var_c_ + (current_b @ self.var_A_.T) + innovations
            beta_paths[:, m, :] = next_b
            current_b = next_b

        # Convert simulated factors to yield curves: Y = B @ H'
        # shape: (n_paths, horizon_months, n_maturities)
        yield_paths = np.matmul(beta_paths, H.T)
        return yield_paths

    def forecast_summary(
        self,
        horizons_months: Dict[str, int] = None,
        quantiles: List[float] = None
    ) -> Dict:
        """
        Returns structured summary matching the contract schema for output/latest.json.
        """
        if horizons_months is None:
            horizons_months = {"3M": 3, "6M": 6, "1Y": 12, "2Y": 24, "3Y": 36, "5Y": 60}
        if quantiles is None:
            quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        max_h = max(horizons_months.values())
        sim_paths = self.simulate(horizon_months=max_h, n_paths=3000)

        summary_horizons = {}
        maturities_list = [float(m) for m in self.maturities_]

        for h_name, h_mo in horizons_months.items():
            step = min(h_mo - 1, max_h - 1)
            # Yield slice at horizon: shape (n_paths, n_maturities)
            slice_y = sim_paths[:, step, :]

            q_dict = {}
            for q in quantiles:
                key = f"q{int(q*100):02d}"
                q_dict[key] = [round(float(v), 3) for v in np.percentile(slice_y, q * 100, axis=0)]

            q_dict["mean"] = [round(float(v), 3) for v in np.mean(slice_y, axis=0)]
            summary_horizons[h_name] = q_dict

        current_vals = [round(float(self.current_curve_.iloc[i]), 3) for i in range(len(self.maturities_))]

        return {
            "maturities": maturities_list,
            "current": current_vals,
            "horizons": summary_horizons,
            "fit_rmse_bp": round(self.fit_rmse_bp, 2)
        }
