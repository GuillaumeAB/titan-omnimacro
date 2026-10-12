"""
quant.models.factors — Macro and Statistical Hybrid Factor Model.
Constructs observable macro factor time series, performs per-asset OLS,
extracts latent statistical factors from residuals via PCA, and fits hybrid exposures.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from quant.config import FACTOR_PROXIES, N_STAT_FACTORS, MIN_HISTORY_WEEKS


def build_factor_returns(weekly_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs weekly macro factor returns according to the factor definitions:
    MKT: log return ACWI
    USD: log return DX-Y.NYB
    RATES: weekly difference of ^TNX (in percentage points)
    OIL: log return BZ=F
    COPPER: log return HG=F
    GOLD: log return GC=F
    CREDIT: log return HYG - log return IEF
    EM: log return EEM - log return ACWI
    """
    wp = weekly_prices.dropna(how="all")
    factors = pd.DataFrame(index=wp.index[1:], dtype=float)

    def _log_ret(ticker: str) -> Optional[pd.Series]:
        if ticker in wp.columns:
            s = wp[ticker].dropna()
            valid = s[s > 0]
            if len(valid) > 1:
                return np.log(valid).diff().iloc[1:]
        return None

    # MKT
    r_acwi = _log_ret("ACWI")
    if r_acwi is not None:
        factors["MKT"] = r_acwi

    # USD
    r_usd = _log_ret("DX-Y.NYB")
    if r_usd is not None:
        factors["USD"] = r_usd

    # RATES: weekly difference in percentage points
    if "^TNX" in wp.columns:
        s_tnx = wp["^TNX"].dropna()
        if len(s_tnx) > 1:
            factors["RATES"] = s_tnx.diff().iloc[1:]

    # OIL
    r_oil = _log_ret("BZ=F")
    if r_oil is None:
        r_oil = _log_ret("CL=F")
    if r_oil is not None:
        factors["OIL"] = r_oil

    # COPPER
    r_cop = _log_ret("HG=F")
    if r_cop is not None:
        factors["COPPER"] = r_cop

    # GOLD
    r_gld = _log_ret("GC=F")
    if r_gld is not None:
        factors["GOLD"] = r_gld

    # CREDIT
    r_hyg = _log_ret("HYG")
    r_ief = _log_ret("IEF")
    if r_hyg is not None and r_ief is not None:
        aligned = pd.concat([r_hyg, r_ief], axis=1).dropna()
        factors["CREDIT"] = aligned.iloc[:, 0] - aligned.iloc[:, 1]

    # EM
    r_eem = _log_ret("EEM")
    if r_eem is not None and r_acwi is not None:
        aligned = pd.concat([r_eem, r_acwi], axis=1).dropna()
        factors["EM"] = aligned.iloc[:, 0] - aligned.iloc[:, 1]

    # Fill any gaps
    return factors.ffill().bfill().dropna(how="all")


class HybridFactorModel:
    """
    Macro + Statistical Factor Model.
    Step 1: OLS regression of each asset on observable macro factors.
    Step 2: SVD / PCA on cross-sectional residuals to extract latent statistical factors.
    Step 3: Secondary regression of residuals on statistical factors to obtain hybrid betas.
    """

    def __init__(
        self,
        n_stat_factors: int = N_STAT_FACTORS,
        min_history: int = MIN_HISTORY_WEEKS
    ):
        self.n_stat_factors = n_stat_factors
        self.min_history = min_history

        self.factor_names_: List[str] = []
        self.factor_returns_: pd.DataFrame = None     # (T, F_macro + F_stat)
        self.betas_: pd.DataFrame = None              # (N, F)
        self.alpha_: pd.Series = None                 # (N,) weekly alpha
        self.resid_vol_: pd.Series = None             # (N,) weekly residual std
        self.resid_: pd.DataFrame = None              # (T, N)
        self.r2_: pd.Series = None                    # (N,) R-squared
        self.stat_explained_var_: List[float] = []
        self.assets_: List[str] = []
        self.dropped_: List[str] = []

    def fit(self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame):
        """
        Fits hybrid factor model on asset returns and macro factor returns.
        """
        # Align index
        common_idx = asset_returns.index.intersection(factor_returns.index)
        Y_all = asset_returns.loc[common_idx]
        F_macro = factor_returns.loc[common_idx]

        macro_cols = list(F_macro.columns)
        T = len(common_idx)

        valid_assets = []
        dropped_assets = []

        # Check history length
        for col in Y_all.columns:
            obs = Y_all[col].dropna()
            if len(obs) >= self.min_history:
                valid_assets.append(col)
            else:
                dropped_assets.append(col)

        self.dropped_ = dropped_assets
        if not valid_assets:
            raise ValueError(f"No assets meet minimum history requirement of {self.min_history} weeks.")

        Y = Y_all[valid_assets]
        N = len(valid_assets)
        self.assets_ = valid_assets

        # Step 1: OLS on macro factors
        macro_betas = np.zeros((N, len(macro_cols)))
        macro_alphas = np.zeros(N)
        macro_resids = pd.DataFrame(index=common_idx, columns=valid_assets, dtype=float)
        r2_macro = np.zeros(N)

        for i, col in enumerate(valid_assets):
            series_y = Y[col]
            valid_mask = series_y.notna() & F_macro.notna().all(axis=1)
            y_sub = series_y[valid_mask].values
            X_sub = F_macro[valid_mask].values

            if len(y_sub) < self.min_history:
                continue

            # OLS with intercept
            X_mat = np.column_stack([np.ones(len(y_sub)), X_sub])
            # Pinverse for numerical stability
            inv = np.linalg.pinv(X_mat.T @ X_mat) @ X_mat.T
            coeffs = inv @ y_sub

            alpha = coeffs[0]
            betas = coeffs[1:]

            macro_alphas[i] = alpha
            macro_betas[i, :] = betas

            fitted = X_mat @ coeffs
            resids = y_sub - fitted
            macro_resids.loc[valid_mask, col] = resids

            # R^2
            ss_tot = np.sum((y_sub - np.mean(y_sub)) ** 2)
            ss_res = np.sum(resids ** 2)
            r2_macro[i] = max(0.0, 1.0 - (ss_res / max(ss_tot, 1e-12)))

        # Step 2: PCA on standardized residuals with complete coverage
        # Find assets with high coverage across the sample
        coverage = macro_resids.notna().sum()
        complete_assets = coverage[coverage >= int(T * 0.85)].index.tolist()

        if len(complete_assets) >= self.n_stat_factors:
            R_pca = macro_resids[complete_assets].ffill().bfill().values
            # Standardize
            std_dev = np.std(R_pca, axis=0)
            std_dev = np.maximum(std_dev, 1e-8)
            R_std = (R_pca - np.mean(R_pca, axis=0)) / std_dev

            # SVD: R_std = U S V^T
            U, S, Vt = np.linalg.svd(R_std, full_matrices=False)
            explained_var = (S ** 2) / np.sum(S ** 2)
            self.stat_explained_var_ = [round(float(v), 4) for v in explained_var[:self.n_stat_factors]]

            # Statistical factors = U[:, :k] * S[:k] / sqrt(T)
            stat_factors = U[:, :self.n_stat_factors] * (S[:self.n_stat_factors] / np.sqrt(T))
            stat_cols = [f"STAT{k+1}" for k in range(self.n_stat_factors)]
            df_stat = pd.DataFrame(stat_factors, index=common_idx, columns=stat_cols)
        else:
            stat_cols = []
            df_stat = pd.DataFrame(index=common_idx)
            self.stat_explained_var_ = []

        # Step 3: Combine Macro + Statistical factors
        full_factors = pd.concat([F_macro, df_stat], axis=1) if not df_stat.empty else F_macro
        self.factor_names_ = list(full_factors.columns)
        self.factor_returns_ = full_factors

        # Regress asset residuals on stat factors (or full factors directly)
        total_factors = len(self.factor_names_)
        final_betas = np.zeros((N, total_factors))
        final_alphas = np.zeros(N)
        final_resids = pd.DataFrame(index=common_idx, columns=valid_assets, dtype=float)
        final_r2 = np.zeros(N)
        final_vol = np.zeros(N)

        F_mat = full_factors.values
        for i, col in enumerate(valid_assets):
            series_y = Y[col]
            valid_mask = series_y.notna() & full_factors.notna().all(axis=1)
            y_sub = series_y[valid_mask].values
            X_sub = F_mat[valid_mask]

            X_mat = np.column_stack([np.ones(len(y_sub)), X_sub])
            inv = np.linalg.pinv(X_mat.T @ X_mat) @ X_mat.T
            coeffs = inv @ y_sub

            final_alphas[i] = coeffs[0]
            final_betas[i, :] = coeffs[1:]

            fitted = X_mat @ coeffs
            resids = y_sub - fitted
            final_resids.loc[valid_mask, col] = resids

            ss_tot = np.sum((y_sub - np.mean(y_sub)) ** 2)
            ss_res = np.sum(resids ** 2)
            final_r2[i] = max(0.0, 1.0 - (ss_res / max(ss_tot, 1e-12)))
            final_vol[i] = float(np.std(resids))

        self.betas_ = pd.DataFrame(final_betas, index=valid_assets, columns=self.factor_names_)
        self.alpha_ = pd.Series(final_alphas, index=valid_assets)
        self.resid_vol_ = pd.Series(final_vol, index=valid_assets)
        self.resid_ = final_resids
        self.r2_ = pd.Series(final_r2, index=valid_assets)

        return self
