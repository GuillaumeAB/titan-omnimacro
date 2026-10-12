"""
OmniMacro Engine - Institutional Multi-Asset Portfolio Construction & Geopolitical Risk Budgeting
Models Global Core + Regional Satellite Allocation (Africa, Middle East, Eastern Europe)
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pydantic import BaseModel, Field

from core.config import INDEX_PROFILES


# Empirical 5-Year Covariance & Correlation Matrix across Global & Regional Asset Classes
REGIONAL_CODES = ["SPX", "NDX", "MSCIWORLD", "STOXX600", "EM_BROAD", "AFRICA", "MIDDLE_EAST", "EASTERN_EUROPE"]

CORRELATION_MATRIX = {
    "SPX":            {"SPX": 1.00, "NDX": 0.92, "MSCIWORLD": 0.96, "STOXX600": 0.78, "EM_BROAD": 0.68, "AFRICA": 0.52, "MIDDLE_EAST": 0.42, "EASTERN_EUROPE": 0.64},
    "NDX":            {"SPX": 0.92, "NDX": 1.00, "MSCIWORLD": 0.91, "STOXX600": 0.72, "EM_BROAD": 0.72, "AFRICA": 0.48, "MIDDLE_EAST": 0.36, "EASTERN_EUROPE": 0.58},
    "MSCIWORLD":      {"SPX": 0.96, "NDX": 0.91, "MSCIWORLD": 1.00, "STOXX600": 0.86, "EM_BROAD": 0.74, "AFRICA": 0.55, "MIDDLE_EAST": 0.46, "EASTERN_EUROPE": 0.68},
    "STOXX600":       {"SPX": 0.78, "NDX": 0.72, "MSCIWORLD": 0.86, "STOXX600": 1.00, "EM_BROAD": 0.69, "AFRICA": 0.58, "MIDDLE_EAST": 0.48, "EASTERN_EUROPE": 0.82},
    "EM_BROAD":       {"SPX": 0.68, "NDX": 0.72, "MSCIWORLD": 0.74, "STOXX600": 0.69, "EM_BROAD": 1.00, "AFRICA": 0.74, "MIDDLE_EAST": 0.58, "EASTERN_EUROPE": 0.71},
    "AFRICA":         {"SPX": 0.52, "NDX": 0.48, "MSCIWORLD": 0.55, "STOXX600": 0.58, "EM_BROAD": 0.74, "AFRICA": 1.00, "MIDDLE_EAST": 0.49, "EASTERN_EUROPE": 0.54},
    "MIDDLE_EAST":    {"SPX": 0.42, "NDX": 0.36, "MSCIWORLD": 0.46, "STOXX600": 0.48, "EM_BROAD": 0.58, "AFRICA": 0.49, "MIDDLE_EAST": 1.00, "EASTERN_EUROPE": 0.45},
    "EASTERN_EUROPE": {"SPX": 0.64, "NDX": 0.58, "MSCIWORLD": 0.68, "STOXX600": 0.82, "EM_BROAD": 0.71, "AFRICA": 0.54, "MIDDLE_EAST": 0.45, "EASTERN_EUROPE": 1.00}
}

ANNUAL_VOLATILITY = {
    "SPX": 0.165,
    "NDX": 0.225,
    "MSCIWORLD": 0.158,
    "STOXX600": 0.170,
    "EM_BROAD": 0.210,
    "AFRICA": 0.245,
    "MIDDLE_EAST": 0.185,
    "EASTERN_EUROPE": 0.235
}

ANNUAL_EXPECTED_RETURN_BASE = {
    "SPX": 0.098,
    "NDX": 0.112,
    "MSCIWORLD": 0.089,
    "STOXX600": 0.084,
    "EM_BROAD": 0.092,
    "AFRICA": 0.078,        # High dividend but chronic FX drag and high TER
    "MIDDLE_EAST": 0.095,   # High dividend, oil cash flow, zero USD-peg FX drag
    "EASTERN_EUROPE": 0.105 # Low starting P/E (8.5x), nearshoring tailwind, reconstruction optionality
}


class PortfolioWeightConfig(BaseModel):
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "SPX": 0.50,
            "NDX": 0.20,
            "STOXX600": 0.15,
            "EM_BROAD": 0.05,
            "MIDDLE_EAST": 0.04,
            "EASTERN_EUROPE": 0.04,
            "AFRICA": 0.02
        }
    )


class PortfolioAnalytics(BaseModel):
    portfolio_name: str
    expected_cagr_5y: float
    total_expected_return_5y: float
    annual_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    ter_weighted_pct: float
    fx_drag_weighted_pct: float
    max_stress_drawdown_5y: float
    marginal_risk_contributions: Dict[str, float]
    geopolitical_scenario_returns: Dict[str, float]
    allocation: Dict[str, float]


class PortfolioEngine:
    """Calculates risk budgeting, diversification ratio, and stress outcomes across global and satellite ETFs."""

    def __init__(self):
        self.codes = REGIONAL_CODES
        self.corr = CORRELATION_MATRIX
        self.vol = ANNUAL_VOLATILITY
        self.ret = ANNUAL_EXPECTED_RETURN_BASE
        self.profiles = INDEX_PROFILES

    def _get_cov_matrix(self) -> np.ndarray:
        n = len(self.codes)
        cov = np.zeros((n, n))
        for i, c1 in enumerate(self.codes):
            for j, c2 in enumerate(self.codes):
                cov[i, j] = self.corr[c1][c2] * self.vol[c1] * self.vol[c2]
        return cov

    def evaluate_portfolio(
        self,
        weights_dict: Dict[str, float],
        name: str = "Custom Allocation"
    ) -> PortfolioAnalytics:
        """Evaluates modern portfolio theory, risk budgeting, and geopolitical scenario resilience."""
        # Normalize weights
        tot = sum(weights_dict.values())
        norm_w = {k: v / tot for k, v in weights_dict.items()} if tot > 0 else {"SPX": 1.0}

        w_vec = np.array([norm_w.get(c, 0.0) for c in self.codes])
        cov = self._get_cov_matrix()

        # Portfolio Volatility
        port_var = float(np.dot(w_vec.T, np.dot(cov, w_vec)))
        port_vol = math.sqrt(max(1e-6, port_var))

        # Weighted Expected Return taking into account TER and FX drag
        net_returns_by_asset = {}
        weighted_ter = 0.0
        weighted_fx_drag = 0.0

        for c in self.codes:
            prof = self.profiles.get(c, {})
            ter = prof.get("ter_annual_pct", 0.002)
            fx = prof.get("fx_drag_annual_pct", 0.0)
            base_r = self.ret[c]
            net_r = base_r - ter + fx
            net_returns_by_asset[c] = net_r
            weighted_ter += norm_w.get(c, 0.0) * ter
            weighted_fx_drag += norm_w.get(c, 0.0) * fx

        port_expected_return = float(sum(w_vec[i] * net_returns_by_asset[self.codes[i]] for i in range(len(self.codes))))

        # 5-Year Compounded Return
        cagr_5y = port_expected_return * 100.0
        total_ret_5y = (((1.0 + port_expected_return) ** 5) - 1.0) * 100.0

        # Sharpe Ratio (Assuming 3.5% risk free rate over 5 years)
        risk_free = 0.035
        sharpe = (port_expected_return - risk_free) / max(0.01, port_vol)

        # Diversification Ratio (Weighted Average Vol / Portfolio Vol)
        weighted_individual_vol = sum(w_vec[i] * self.vol[self.codes[i]] for i in range(len(self.codes)))
        div_ratio = weighted_individual_vol / max(1e-4, port_vol)

        # Marginal Risk Contributions (MRC)
        # MCR_i = (Cov * w)_i / port_vol
        cov_w = np.dot(cov, w_vec)
        mrc_dict = {}
        for i, c in enumerate(self.codes):
            if w_vec[i] > 0.001:
                # Percentage contribution to total variance = w_i * cov_w_i / port_var
                pct_contrib = (w_vec[i] * cov_w[i]) / max(1e-6, port_var)
                mrc_dict[c] = round(pct_contrib * 100.0, 1)

        # Stress Test Scenarios across asset classes
        scenario_impacts = {
            "Baseline Soft-Landing": round(total_ret_5y, 1),
            "AI Supercycle Boom": round(
                total_ret_5y + (norm_w.get("NDX", 0) * 35.0) + (norm_w.get("SPX", 0) * 18.0) - (norm_w.get("MIDDLE_EAST", 0) * 5.0),
                1
            ),
            "Taiwan Strait Blockade Shock": round(
                total_ret_5y - (norm_w.get("NDX", 0) * 38.0) - (norm_w.get("SPX", 0) * 22.0) + 
                (norm_w.get("EASTERN_EUROPE", 0) * 12.0) + (norm_w.get("MIDDLE_EAST", 0) * 8.0),
                1
            ),
            "Middle East Hormuz Oil Shock ($135)": round(
                total_ret_5y + (norm_w.get("MIDDLE_EAST", 0) * 45.0) - (norm_w.get("STOXX600", 0) * 20.0) - 
                (norm_w.get("SPX", 0) * 12.0) + (norm_w.get("AFRICA", 0) * 15.0),
                1
            ),
            "Ukraine Ceasefire & EU Reconstruction": round(
                total_ret_5y + (norm_w.get("EASTERN_EUROPE", 0) * 48.0) + (norm_w.get("STOXX600", 0) * 18.0) - 
                (norm_w.get("MIDDLE_EAST", 0) * 10.0),
                1
            )
        }

        # Maximum Stress Drawdown
        max_dd = 0.28 * (norm_w.get("NDX", 0) * 1.4 + norm_w.get("SPX", 0) * 1.0 + norm_w.get("AFRICA", 0) * 1.3)
        max_dd = max_dd / div_ratio  # Diversification reduces drawdown

        return PortfolioAnalytics(
            portfolio_name=name,
            expected_cagr_5y=round(cagr_5y, 2),
            total_expected_return_5y=round(total_ret_5y, 1),
            annual_volatility=round(port_vol * 100, 2),
            sharpe_ratio=round(sharpe, 2),
            diversification_ratio=round(div_ratio, 2),
            ter_weighted_pct=round(weighted_ter * 100, 3),
            fx_drag_weighted_pct=round(weighted_fx_drag * 100, 3),
            max_stress_drawdown_5y=round(max_dd * 100, 1),
            marginal_risk_contributions=mrc_dict,
            geopolitical_scenario_returns=scenario_impacts,
            allocation={k: round(v * 100, 1) for k, v in norm_w.items() if v > 0.001}
        )
